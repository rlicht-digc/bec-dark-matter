#!/usr/bin/env python3
"""
match_photometry.py — Match Vrot-only galaxies with S4G photometry + ALFALFA HI masses
to compute baryonic velocity components (Vdisk, Vgas) for RAR analysis.

Strategy:
  1. S4G Salo+2015 decomposition → disk scale length (hr3) + disk flux fraction (f3)
  2. S4G Sheth+2010 → distance (Dmean), total stellar mass (logM*)
  3. Disk mass = f3 × 10^logM* (disk flux fraction × total stellar mass)
  4. Freeman (1970) exponential disk → Vdisk(R)
  5. ALFALFA Haynes+2018 → total HI mass
  6. Exponential HI disk (scale length ~ 2× stellar) → Vgas(R)

For galaxies without ALFALFA match, we still compute stellar-only gbar.
"""

import os
import re
import numpy as np
from scipy.special import i0, i1, k0, k1

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZIER_DIR = os.path.join(BASE, 'data', 'vizier_catalogs')
ALFALFA_DIR = os.path.join(BASE, 'data', 'alfalfa')

G_PC = 4.302e-3  # pc (km/s)^2 / Msun
PC_PER_KPC = 1000
ARCSEC_PER_RAD = 206265.0
HI_GAS_FACTOR = 1.33  # Correction for helium


def _normalize_name(name):
    name = name.strip().strip('"')
    name = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
                  r'\1', name, flags=re.IGNORECASE)
    name = re.sub(r'^N(\d)', r'NGC\1', name)
    name = re.sub(r'^U(\d)', r'UGC\1', name)
    return name.upper().strip()


def freeman_vdisk(R_kpc, h_kpc, Mdisk_Msun):
    """Compute Vdisk(R) for an exponential disk using Freeman (1970).

    V²(R) = 4πGΣ₀ h y² [I₀(y)K₀(y) - I₁(y)K₁(y)]
    where y = R/(2h), Σ₀ = M/(2πh²)
    """
    if Mdisk_Msun <= 0 or h_kpc <= 0:
        return np.zeros_like(R_kpc)

    Sigma0 = Mdisk_Msun / (2 * np.pi * (h_kpc * PC_PER_KPC)**2)  # Msun/pc²
    y = np.asarray(R_kpc) / (2 * h_kpc)
    y = np.clip(y, 1e-6, 50)

    bessel = i0(y) * k0(y) - i1(y) * k1(y)
    h_pc = h_kpc * PC_PER_KPC
    V2 = 4 * np.pi * G_PC * Sigma0 * h_pc * y**2 * bessel
    return np.sqrt(np.maximum(V2, 0))


def exponential_vgas(R_kpc, h_gas_kpc, Mgas_Msun):
    """Compute Vgas(R) for an exponential gas disk.

    Same Freeman formula but for the gas component.
    M_gas = 1.33 × M_HI (helium correction already applied to Mgas_Msun).
    """
    return freeman_vdisk(R_kpc, h_gas_kpc, Mgas_Msun)


def _hms_to_deg(hms):
    parts = hms.strip().split()
    if len(parts) < 3:
        return None
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return 15.0 * (h + m/60 + s/3600)


def _dms_to_deg(dms):
    parts = dms.strip().split()
    if len(parts) < 3:
        return None
    sign = -1 if parts[0].startswith('-') else 1
    d, m, s = abs(float(parts[0])), float(parts[1]), float(parts[2])
    return sign * (d + m/60 + s/3600)


def load_s4g_data():
    """Load S4G decompositions and properties.

    Returns:
        s4g_disks: dict of normalized_name -> {f3, mag3, hr3_arcsec}
        s4g_props: dict of normalized_name -> {dist_Mpc, logMstar, ell, ra, dec}
    """
    # Decompositions
    s4g_disks = {}
    decomp_fp = os.path.join(VIZIER_DIR, 's4g_salo2015_decomp.tsv')
    if os.path.exists(decomp_fp):
        with open(decomp_fp) as f:
            header = None
            for line in f:
                if line.startswith('#') or line.startswith('-') or not line.strip():
                    continue
                parts = line.split('\t')
                if header is None:
                    header = [h.strip() for h in parts]
                    continue
                if len(parts) < len(header):
                    continue
                row = {h: parts[i].strip() for i, h in enumerate(header)}
                nn = _normalize_name(row.get('Name', ''))
                fn = row.get('Fn', '').strip()
                if fn != 'expdisk':
                    continue
                try:
                    f3 = float(row.get('f3', '')) if row.get('f3', '').strip() else 0
                    mag3 = float(row.get('mag3', '')) if row.get('mag3', '').strip() else None
                    hr3 = float(row.get('hr3', '')) if row.get('hr3', '').strip() else None
                except ValueError:
                    continue
                if hr3 is None:
                    continue
                if nn not in s4g_disks or f3 > s4g_disks[nn]['f3']:
                    s4g_disks[nn] = {'f3': f3, 'mag3': mag3, 'hr3_arcsec': hr3}

    # Properties
    s4g_props = {}
    props_fp = os.path.join(VIZIER_DIR, 's4g_sheth2010_props.tsv')
    if os.path.exists(props_fp):
        with open(props_fp) as f:
            header = None
            for line in f:
                if line.startswith('#') or line.startswith('-') or not line.strip():
                    continue
                parts = line.split('\t')
                if header is None:
                    header = [h.strip() for h in parts]
                    continue
                if len(parts) < len(header):
                    continue
                row = {h: parts[i].strip() for i, h in enumerate(header)}
                nn = _normalize_name(row.get('Name', ''))
                try:
                    dist = float(row.get('Dmean', '')) if row.get('Dmean', '').strip() else None
                    logMstar = float(row.get('logM*', '')) if row.get('logM*', '').strip() else None
                    ell = float(row.get('ell', '')) if row.get('ell', '').strip() else None
                    ra = float(row.get('RAJ2000', ''))
                    dec = float(row.get('DEJ2000', ''))
                except (ValueError, KeyError):
                    continue
                if dist is not None and dist > 0:
                    inc = np.degrees(np.arccos(1 - ell)) if ell is not None else None
                    s4g_props[nn] = {
                        'dist_Mpc': dist, 'logMstar': logMstar,
                        'ell': ell, 'inc_deg': inc,
                        'ra': ra, 'dec': dec,
                    }

    return s4g_disks, s4g_props


def load_alfalfa():
    """Load ALFALFA HI catalog with coordinates for cross-matching.

    Returns arrays: ra, dec, logMHI
    """
    fp = os.path.join(ALFALFA_DIR, 'alfalfa_alpha100_haynes2018.tsv')
    if not os.path.exists(fp):
        return np.array([]), np.array([]), np.array([])

    ras, decs, logMHIs = [], [], []
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

            ra = _hms_to_deg(row.get('RAJ2000', ''))
            dec = _dms_to_deg(row.get('DEJ2000', ''))
            try:
                logMHI = float(row.get('logMHI', ''))
            except ValueError:
                continue

            if ra is not None and dec is not None:
                ras.append(ra)
                decs.append(dec)
                logMHIs.append(logMHI)

    return np.array(ras), np.array(decs), np.array(logMHIs)


def match_alfalfa(target_ra, target_dec, alf_ra, alf_dec, alf_logMHI,
                  match_radius_arcmin=3.0):
    """Cross-match a single target with ALFALFA by coordinates."""
    if len(alf_ra) == 0:
        return None
    cosdec = np.cos(np.radians(target_dec))
    dra = (alf_ra - target_ra) * cosdec
    ddec = alf_dec - target_dec
    sep = np.sqrt(dra**2 + ddec**2)
    idx = np.argmin(sep)
    if sep[idx] < match_radius_arcmin / 60.0:
        return alf_logMHI[idx]
    return None


def compute_baryonic_components(galaxies, verbose=True):
    """Add Vdisk and Vgas to Vrot-only galaxies using S4G + ALFALFA.

    Modifies galaxies dict in-place: sets Vgas, Vdisk, Vbul arrays
    and has_mass_model=True for matched galaxies.

    Returns dict of match statistics.
    """
    s4g_disks, s4g_props = load_s4g_data()
    alf_ra, alf_dec, alf_logMHI = load_alfalfa()

    stats = {'s4g_matched': 0, 'alfalfa_matched': 0, 'stellar_only': 0, 'skipped': 0}

    for name, g in galaxies.items():
        # Skip galaxies that already have mass models
        if g.get('has_mass_model', False) or g['source'] == 'SPARC':
            continue

        nn = _normalize_name(name)

        # Check S4G match
        if nn not in s4g_disks or nn not in s4g_props:
            stats['skipped'] += 1
            continue

        disk = s4g_disks[nn]
        props = s4g_props[nn]
        dist = props['dist_Mpc']

        # Disk scale length: arcsec → kpc
        h_kpc = disk['hr3_arcsec'] * dist / ARCSEC_PER_RAD * 1e3

        # Disk mass from S4G total M* × disk flux fraction
        if props.get('logMstar') is not None:
            M_total = 10**props['logMstar']
            M_disk = disk['f3'] * M_total
        else:
            stats['skipped'] += 1
            continue

        R_kpc = g['R_kpc']
        Vdisk = freeman_vdisk(R_kpc, h_kpc, M_disk)

        # Try ALFALFA match for Vgas
        logMHI = match_alfalfa(props['ra'], props['dec'], alf_ra, alf_dec, alf_logMHI)

        if logMHI is not None:
            M_gas = HI_GAS_FACTOR * 10**logMHI
            h_gas_kpc = 2.0 * h_kpc  # HI disk typically ~2× stellar scale length
            Vgas = exponential_vgas(R_kpc, h_gas_kpc, M_gas)
            stats['alfalfa_matched'] += 1
        else:
            Vgas = np.zeros_like(R_kpc)
            stats['stellar_only'] += 1

        g['Vdisk'] = Vdisk
        g['Vgas'] = Vgas
        g['Vbul'] = np.zeros_like(R_kpc)  # Bulge contribution not computed
        g['has_mass_model'] = True
        g['dist_Mpc'] = dist
        if props.get('inc_deg') is not None:
            g['inc_deg'] = props['inc_deg']
        g['mass_model_source'] = 'S4G+ALFALFA' if logMHI else 'S4G_stellar'

        stats['s4g_matched'] += 1

    if verbose:
        print(f"  S4G matched: {stats['s4g_matched']} galaxies")
        print(f"    with ALFALFA HI: {stats['alfalfa_matched']}")
        print(f"    stellar only: {stats['stellar_only']}")
        print(f"  Unmatched (no S4G): {stats['skipped']}")

    return stats


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_extended_rar import load_all, build_rar

    print("=" * 70)
    print("Photometry matching + RAR extension")
    print("=" * 70)

    galaxies = load_all(include_vizier=True)

    print("\nMatching with S4G + ALFALFA...")
    stats = compute_baryonic_components(galaxies)

    # Updated counts
    n_rar = sum(1 for g in galaxies.values()
                if g.get('has_mass_model', False) or g['source'] == 'SPARC')
    print(f"\nTotal RAR-usable galaxies: {n_rar}")

    # Build RAR
    print("\n--- Extended RAR ---")
    log_gobs, log_gbar, names, sources = build_rar(galaxies)
    print(f"Total RAR points: {len(log_gobs)}")
    if len(log_gobs) > 0:
        print(f"  gobs range: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")
        print(f"  gbar range: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")

        for src in sorted(set(sources)):
            mask = sources == src
            print(f"  {src}: {mask.sum()} RAR points from {len(set(names[mask]))} galaxies")

    # Breakdown by mass model source
    mm_sources = {}
    for name, g in galaxies.items():
        mms = g.get('mass_model_source', g['source'] if g['source'] == 'SPARC' else 'none')
        mm_sources.setdefault(mms, []).append(name)
    print("\n--- Mass model sources ---")
    for mms, ns in sorted(mm_sources.items()):
        print(f"  {mms}: {len(ns)} galaxies")
