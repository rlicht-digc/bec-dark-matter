#!/usr/bin/env python3
"""
match_bouquin_photometry.py — Compute Vdisk from Bouquin+2018 3.6μm radial
surface brightness profiles for Vrot-only galaxies.

Strategy:
  1. Bouquin+2018 table3: μ₃.₆(R) radial profiles at 6" steps
  2. Bouquin+2018 table1: distances, morphological types
  3. Convert μ₃.₆ → Σ★(R) using M/L₃.₆ = 0.5 (McGaugh & Schombert 2014)
  4. Compute Vdisk(R) numerically from Σ★(R) using Casertano (1983)
  5. Also add ALFALFA Vgas if available

Advantage over S4G parametric approach (match_photometry.py):
  - Uses actual measured μ(R) at each radius, not exponential disk assumption
  - Handles non-exponential profiles (bars, rings, breaks) naturally
  - Should give lower scatter in RAR residuals

Reference: Casertano 1983, MNRAS 203, 735 — disk rotation from arbitrary Σ(R)
"""

import os
import re
import numpy as np
from scipy.special import ellipk, ellipe

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZIER_DIR = os.path.join(BASE, 'data', 'vizier_catalogs')
ALFALFA_DIR = os.path.join(BASE, 'data', 'alfalfa')

G_CGS = 6.674e-8       # cm^3 g^-1 s^-2
G_PC = 4.302e-3         # pc (km/s)^2 / Msun
PC_CM = 3.086e18        # cm per pc
KPC_CM = 3.086e21       # cm per kpc
ARCSEC_PER_RAD = 206265.0
MSUN_AB_36 = 6.02       # AB magnitude of Sun at 3.6μm (Willmer 2018; Vega = 3.24)
ML_RATIO_36 = 0.5       # Stellar M/L at 3.6μm (McGaugh & Schombert 2014)
HI_GAS_FACTOR = 1.33    # Helium correction


def _normalize_name(name):
    name = name.strip().strip('"')
    name = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
                  r'\1', name, flags=re.IGNORECASE)
    name = re.sub(r'^N(\d)', r'NGC\1', name)
    name = re.sub(r'^U(\d)', r'UGC\1', name)
    return name.upper().strip()


def mu36_to_sigma(mu36, ml_ratio=ML_RATIO_36):
    """Convert 3.6μm AB surface brightness to stellar surface mass density.

    μ in mag/arcsec² → Σ★ in Msun/pc²

    Conversion:
      I [L_sun/pc²] = 10^(0.4 * (M_sun,3.6 - μ + 21.572))
      where 21.572 = 2.5*log10(arcsec²/sr × L_sun/pc² conversion)
      Σ★ = (M/L) × I
    """
    log_intensity = 0.4 * (MSUN_AB_36 - mu36 + 21.572)
    return ml_ratio * 10.0**log_intensity


def vdisk_from_sigma(R_pc, Sigma_Msun_pc2):
    """Compute circular velocity from arbitrary Σ(R) for a thin disk.

    Uses the Casertano (1983) / Binney & Tremaine approach:
    For a thin ring of mass dM at radius Rp, contribution to V² at R:
      dV² = (G dM / π) × [K(k)/(R+Rp) + E(k)×(R²-Rp²)/((R-Rp)²×(R+Rp))]
    where k² = 4RRp/(R+Rp)².

    Args:
        R_pc: radii in pc (1D array, monotonically increasing)
        Sigma_Msun_pc2: surface mass density at each R in Msun/pc² (same length)

    Returns:
        V in km/s at each R
    """
    n = len(R_pc)
    if n < 3:
        return np.zeros(n)

    V2 = np.zeros(n)

    for i in range(n):
        R = R_pc[i]
        if R <= 0:
            continue

        V2_val = 0.0
        for j in range(n - 1):
            R1 = max(R_pc[j], 1e-3)
            R2 = R_pc[j + 1]
            Rp = 0.5 * (R1 + R2)
            dR = R2 - R1
            Sp = 0.5 * (Sigma_Msun_pc2[j] + Sigma_Msun_pc2[j + 1])
            if Sp <= 0 or dR <= 0:
                continue

            k2 = 4.0 * R * Rp / (R + Rp)**2
            k2 = min(k2, 0.9999999)

            K_ell = ellipk(k2)
            E_ell = ellipe(k2)

            dM = 2 * np.pi * Rp * Sp * dR
            sum_R = R + Rp
            diff_R2 = (R - Rp)**2
            if diff_R2 < 1e-6:
                diff_R2 = 1e-6  # regularize near R=Rp

            contrib = (G_PC / np.pi) * dM * (K_ell / sum_R +
                       E_ell * (R**2 - Rp**2) / (diff_R2 * sum_R))
            V2_val += contrib

        V2[i] = V2_val

    return np.sqrt(np.maximum(V2, 0))


def vdisk_from_sigma_simple(R_kpc, Sigma_Msun_pc2):
    """Simpler Vdisk using cumulative mass approximation (for validation).

    At each radius, V²(R) ≈ G M(<R) / R where M(<R) is from trapezoidal integration.
    This underestimates by ~15% for an exponential disk but gives the right order.
    """
    R_pc = R_kpc * 1000.0
    n = len(R_pc)
    V = np.zeros(n)

    for i in range(1, n):
        # Integrate Σ in rings from 0 to R[i]
        M_enc = 0.0
        for j in range(i):
            R1 = R_pc[j]
            R2 = R_pc[j + 1] if j + 1 <= i else R_pc[j]
            Rmid = 0.5 * (R1 + R2)
            dR = R2 - R1
            Smid = 0.5 * (Sigma_Msun_pc2[j] + Sigma_Msun_pc2[min(j + 1, n - 1)])
            M_enc += 2 * np.pi * Rmid * Smid * dR

        if R_pc[i] > 0:
            V[i] = np.sqrt(G_PC * M_enc / R_pc[i])

    return V


def load_bouquin_profiles():
    """Load Bouquin+2018 table3 radial SB profiles.

    Returns: dict of name -> list of {sma_arcsec, mu36, ell}
    """
    fp = os.path.join(VIZIER_DIR, 'bouquin2018_table3.tsv')
    if not os.path.exists(fp):
        return {}

    profiles = {}
    with open(fp) as f:
        header = None
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            parts = line.split('\t')
            if header is None:
                header = [h.strip() for h in parts]
                continue
            if len(parts) < len(header):
                continue
            row = {h: parts[i].strip().strip('"') for i, h in enumerate(header)}
            name = row.get('Name', '').strip()
            if not name:
                continue
            try:
                sma = float(row.get('sma', ''))
                mu36 = float(row.get('mu36corr', '')) if row.get('mu36corr', '').strip() else None
                ell = float(row.get('ell', '')) if row.get('ell', '').strip() else 0
            except ValueError:
                continue
            if name not in profiles:
                profiles[name] = []
            profiles[name].append({'sma_arcsec': sma, 'mu36': mu36, 'ell': ell})

    return profiles


def load_bouquin_properties():
    """Load Bouquin+2018 table1 galaxy properties.

    Returns: dict of name -> {dist_Mpc, ra, dec, T, mag36}
    """
    fp = os.path.join(VIZIER_DIR, 'bouquin2018_table1.tsv')
    if not os.path.exists(fp):
        return {}

    props = {}
    with open(fp) as f:
        header = None
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            parts = line.split('\t')
            if header is None:
                header = [h.strip() for h in parts]
                continue
            if len(parts) < len(header):
                continue
            row = {h: parts[i].strip().strip('"') for i, h in enumerate(header)}
            name = row.get('Name', '').strip()
            if not name:
                continue
            try:
                dist = float(row.get('Dist', '')) if row.get('Dist', '').strip() else None
                ra = float(row.get('RAJ2000', '')) if row.get('RAJ2000', '').strip() else None
                dec = float(row.get('DEJ2000', '')) if row.get('DEJ2000', '').strip() else None
                T = float(row.get('T', '')) if row.get('T', '').strip() else None
            except ValueError:
                continue
            if dist is not None and dist > 0:
                props[name] = {'dist_Mpc': dist, 'ra': ra, 'dec': dec, 'T': T}

    return props


def compute_vdisk_from_profile(profile_pts, dist_Mpc, R_kpc_target):
    """Compute Vdisk at target radii from a Bouquin+2018 SB profile.

    Args:
        profile_pts: list of {sma_arcsec, mu36, ell} dicts
        dist_Mpc: distance in Mpc
        R_kpc_target: array of radii (kpc) at which to evaluate Vdisk

    Returns:
        Vdisk in km/s at each target radius, or None if insufficient data
    """
    # Filter to valid points
    valid = [(p['sma_arcsec'], p['mu36']) for p in profile_pts
             if p['mu36'] is not None and p['sma_arcsec'] > 0]
    if len(valid) < 3:
        return None

    valid.sort(key=lambda x: x[0])
    sma_arcsec = np.array([v[0] for v in valid])
    mu36 = np.array([v[1] for v in valid])

    # Convert angular → physical radius
    R_kpc = sma_arcsec * dist_Mpc / ARCSEC_PER_RAD * 1e3
    R_pc = R_kpc * 1000.0

    # Convert SB → surface mass density
    Sigma = mu36_to_sigma(mu36)

    # Compute V using proper thin-disk calculation (Casertano 1983)
    R_pc = R_kpc * 1000.0
    V_disk = vdisk_from_sigma(R_pc, Sigma)

    # Interpolate to target radii
    V_target = np.interp(R_kpc_target, R_kpc, V_disk,
                         left=0, right=V_disk[-1] if len(V_disk) > 0 else 0)

    return V_target


def compute_baryonic_bouquin(galaxies, verbose=True):
    """Add Vdisk from Bouquin+2018 profiles to Vrot-only galaxies.

    Modifies galaxies dict in-place. Only processes galaxies that:
    1. Don't already have a mass model
    2. Are not SPARC (already have mass models)
    3. Have a match in Bouquin+2018

    Returns dict of match statistics.
    """
    bouquin_profiles = load_bouquin_profiles()
    bouquin_props = load_bouquin_properties()

    # Also load ALFALFA for Vgas
    from match_photometry import load_alfalfa, match_alfalfa, exponential_vgas

    alf_ra, alf_dec, alf_logMHI = load_alfalfa()

    # Build normalized name lookup
    bouquin_norm = {}
    for name in bouquin_profiles:
        nn = _normalize_name(name)
        bouquin_norm[nn] = name

    stats = {'bouquin_matched': 0, 'alfalfa_matched': 0,
             'stellar_only': 0, 'skipped': 0}

    for name, g in galaxies.items():
        if g.get('has_mass_model', False) or g['source'] == 'SPARC':
            continue

        nn = _normalize_name(name)
        if nn not in bouquin_norm:
            stats['skipped'] += 1
            continue

        bname = bouquin_norm[nn]
        if bname not in bouquin_props:
            stats['skipped'] += 1
            continue

        props = bouquin_props[bname]
        profile = bouquin_profiles[bname]
        dist = props['dist_Mpc']

        R_kpc = g['R_kpc']

        # Compute Vdisk from Bouquin profile
        Vdisk = compute_vdisk_from_profile(profile, dist, R_kpc)
        if Vdisk is None:
            stats['skipped'] += 1
            continue

        # Try ALFALFA for Vgas
        logMHI = None
        if props.get('ra') is not None and props.get('dec') is not None:
            logMHI = match_alfalfa(props['ra'], props['dec'],
                                   alf_ra, alf_dec, alf_logMHI)

        if logMHI is not None:
            M_gas = HI_GAS_FACTOR * 10**logMHI
            # Approximate gas disk: use 2× outermost profile radius as gas scale
            valid_prof = [p for p in profile if p['mu36'] is not None]
            if valid_prof:
                max_sma = max(p['sma_arcsec'] for p in valid_prof)
                h_gas_kpc = max_sma * dist / ARCSEC_PER_RAD * 1e3 * 0.5
                h_gas_kpc = max(h_gas_kpc, 1.0)
            else:
                h_gas_kpc = 5.0  # default
            Vgas = exponential_vgas(R_kpc, h_gas_kpc, M_gas)
            stats['alfalfa_matched'] += 1
        else:
            Vgas = np.zeros_like(R_kpc)
            stats['stellar_only'] += 1

        g['Vdisk'] = Vdisk
        g['Vgas'] = Vgas
        g['Vbul'] = np.zeros_like(R_kpc)
        g['has_mass_model'] = True
        g['dist_Mpc'] = dist
        g['mass_model_source'] = 'Bouquin+ALFALFA' if logMHI else 'Bouquin_stellar'
        stats['bouquin_matched'] += 1

    if verbose:
        print(f"  Bouquin matched: {stats['bouquin_matched']} galaxies")
        print(f"    with ALFALFA HI: {stats['alfalfa_matched']}")
        print(f"    stellar only: {stats['stellar_only']}")
        print(f"  Unmatched: {stats['skipped']}")

    return stats


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_extended_rar import load_all, build_rar

    print("=" * 70)
    print("Bouquin+2018 profile matching + Vdisk computation")
    print("=" * 70)

    galaxies = load_all(include_vizier=True)

    print("\nMatching with Bouquin+2018 profiles + ALFALFA...")
    stats = compute_baryonic_bouquin(galaxies)

    n_rar = sum(1 for g in galaxies.values()
                if g.get('has_mass_model', False) or g['source'] == 'SPARC')
    print(f"\nTotal RAR-usable galaxies: {n_rar}")

    print("\n--- Extended RAR ---")
    log_gobs, log_gbar, names, sources = build_rar(galaxies)
    print(f"Total RAR points: {len(log_gobs)}")

    if len(log_gobs) > 0:
        # Scatter by mass model source
        def rar_function(log_gbar, a0=1.2e-10):
            gbar = 10.0**log_gbar
            gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
            return np.log10(gobs)

        log_res = log_gobs - rar_function(log_gbar)

        mm_scatter = {}
        for i in range(len(log_gobs)):
            gname = names[i]
            g = galaxies.get(gname)
            if g:
                mms = g.get('mass_model_source', g['source'] if g['source'] == 'SPARC' else 'unknown')
            else:
                mms = 'unknown'
            if mms not in mm_scatter:
                mm_scatter[mms] = []
            mm_scatter[mms].append(log_res[i])

        print("\nRAR scatter by mass model source:")
        print(f"  {'Source':25s} {'N_pts':>7s} {'σ(res)':>8s} {'<res>':>8s}")
        for mms, residuals in sorted(mm_scatter.items()):
            r = np.array(residuals)
            print(f"  {mms:25s} {len(r):7d} {np.std(r):8.4f} {np.mean(r):+8.4f}")
