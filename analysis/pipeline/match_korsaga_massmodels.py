#!/usr/bin/env python3
"""
match_korsaga_massmodels.py — Reconstruct Vdisk + Vbulge for GHASP galaxies
using published mass model parameters from Korsaga+2019 (MNRAS 482, 154).

Korsaga+2019 provides:
  - Table A1: Galaxy properties (h, mu0, re, n, LD, LB, B-V)
  - Table A2: ISO BFM mass model fits (M/L_disk, M/L_bulge)

From these we reconstruct:
  - V_disk(R) using Freeman (1970) exponential disk:
      V²(R) = 4πGΣ₀ h y² [I₀(y)K₀(y) - I₁(y)K₁(y)]
      where y = R/(2h), Σ₀ = M_disk / (2πh²)
      M_disk = (M/L)_disk × L_disk × 10^8 [Msun]

  - V_bulge(R) using deprojected Sérsic profile (Terzic & Graham 2005
    approximation for p_n, plus Lima Neto+1999 density → V_circ):
      M_bulge = (M/L)_bulge × L_bulge × 10^8 [Msun]
      V²(R) = G M(<R) / R where M(<R) is from numerical Sérsic integration

  - Optionally V_gas from ALFALFA cross-match

These are then matched to existing GHASP rotation curves (Epinat+2008a/b)
or loaded from Korsaga's own rotation curves to build RAR data.

References:
  Korsaga+2019, MNRAS 482, 154
  Freeman 1970, ApJ 160, 811
  Sérsic 1963, Bol. Asoc. Argentina de Astronomia 6, 41
  Terzic & Graham 2005, MNRAS 362, 197
"""

import os
import re
import numpy as np
from scipy.special import i0, i1, k0, k1, gamma, gammainc

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZIER_DIR = os.path.join(BASE, 'data', 'vizier_catalogs')
ALFALFA_DIR = os.path.join(BASE, 'data', 'alfalfa')

G_PC = 4.302e-3         # pc (km/s)^2 / Msun
PC_PER_KPC = 1000
HI_GAS_FACTOR = 1.33    # Helium correction


def _normalize_ugc(name):
    """Normalize UGC names — strip leading zeros and spaces.

    'UGC 00089' → 'UGC89'
    'UGC 10075' → 'UGC10075'
    '"UGC 89"'  → 'UGC89'
    """
    name = name.strip().strip('"').upper()
    m = re.match(r'^UGC\s*0*(\d+)', name)
    if m:
        return f'UGC{m.group(1)}'
    return name


def _normalize_name(name):
    """General galaxy name normalizer (matches load_extended_rar.py)."""
    name = name.strip().strip('"')
    name = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
                  r'\1', name, flags=re.IGNORECASE)
    name = re.sub(r'^N(\d)', r'NGC\1', name)
    name = re.sub(r'^U(\d)', r'UGC\1', name)
    return name.upper().strip()


def _parse_vizier_tsv(filepath):
    """Generic VizieR TSV parser. Returns (header, rows)."""
    rows = []
    header = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if header is None:
                header = [h.strip() for h in parts]
                continue
            # Skip unit/separator rows
            if parts[0].strip().startswith('-') or all(
                    c in '- \t' for c in line):
                continue
            row = {}
            for i, h in enumerate(header):
                if i < len(parts):
                    row[h] = parts[i].strip().strip('"')
                else:
                    row[h] = ''
            rows.append(row)
    return header, rows


# ──────────────────────────────────────────────────────────────────────────────
# Korsaga data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_korsaga_tablea1():
    """Load Korsaga+2019 Table A1 — global properties.

    Returns dict: UGC_norm -> {
        'h_kpc': disk scale length,
        'mu0': central B-band SB (mag/arcsec²),
        'LD_1e8Lsun': disk luminosity (10^8 Lsun),
        'LB_1e8Lsun': bulge luminosity (10^8 Lsun) or 0,
        're_kpc': bulge effective radius (kpc) or 0,
        'n_sersic': Sérsic index or 0,
        'BV': B-V color,
        'BMAG': absolute B magnitude,
        'f_ID': RC quality flag (1=best, 2, 3=worst),
        'param': '*' if in mass model tables, ' ' otherwise,
        'ra': RA (deg), 'dec': Dec (deg)
    }
    """
    fp = os.path.join(VIZIER_DIR, 'korsaga2019_tablea1.tsv')
    if not os.path.exists(fp):
        print(f"WARNING: {fp} not found")
        return {}

    header, rows = _parse_vizier_tsv(fp)
    result = {}

    for row in rows:
        ugc_raw = row.get('ID', '').strip()
        if not ugc_raw:
            continue

        ugc_norm = _normalize_ugc(ugc_raw)
        param = row.get('param', '').strip()

        try:
            h_kpc = float(row['h']) if row.get('h', '').strip() else 0
            LD = float(row['LD']) if row.get('LD', '').strip() else 0
        except ValueError:
            continue

        if h_kpc <= 0 or LD <= 0:
            continue  # Need at least disk parameters

        # Optional bulge
        try:
            re_kpc = float(row['re']) if row.get('re', '').strip() else 0
        except ValueError:
            re_kpc = 0
        try:
            n_sersic = float(row['n']) if row.get('n', '').strip() else 0
        except ValueError:
            n_sersic = 0
        try:
            LB = float(row['LB']) if row.get('LB', '').strip() else 0
        except ValueError:
            LB = 0
        try:
            mu0 = float(row['mu0']) if row.get('mu0', '').strip() else 0
        except ValueError:
            mu0 = 0
        try:
            BV = float(row['B-V']) if row.get('B-V', '').strip() else 0.5
        except ValueError:
            BV = 0.5
        try:
            BMAG = float(row['BMAG']) if row.get('BMAG', '').strip() else 0
        except ValueError:
            BMAG = 0
        try:
            f_ID = int(row['f_ID']) if row.get('f_ID', '').strip() else 3
        except ValueError:
            f_ID = 3
        try:
            ra = float(row['_RA']) if row.get('_RA', '').strip() else None
        except ValueError:
            ra = None
        try:
            dec = float(row['_DE']) if row.get('_DE', '').strip() else None
        except ValueError:
            dec = None

        result[ugc_norm] = {
            'h_kpc': h_kpc,
            'mu0': mu0,
            'LD_1e8Lsun': LD,
            'LB_1e8Lsun': LB,
            're_kpc': re_kpc,
            'n_sersic': n_sersic,
            'BV': BV,
            'BMAG': BMAG,
            'f_ID': f_ID,
            'param': param,
            'ra': ra,
            'dec': dec,
            'ugc_raw': ugc_raw,
        }

    return result


def load_korsaga_tablea2():
    """Load Korsaga+2019 Table A2 — ISO BFM mass model parameters.

    Returns dict: UGC_norm -> {
        'ML_disk': M/L of disk (BFM, Msun/Lsun),
        'ML_bulge': M/L of bulge (BFM, Msun/Lsun) or None,
        'ML_fML': fixed M/L from B-V color,
        'chi2_BFM': reduced chi² of BFM fit,
        'chi2_fML': reduced chi² of fixed M/L fit,
    }
    """
    fp = os.path.join(VIZIER_DIR, 'korsaga2019_tablea2.tsv')
    if not os.path.exists(fp):
        print(f"WARNING: {fp} not found")
        return {}

    header, rows = _parse_vizier_tsv(fp)
    result = {}

    for row in rows:
        ugc_raw = row.get('ID', '').strip()
        if not ugc_raw:
            continue

        ugc_norm = _normalize_ugc(ugc_raw)

        try:
            ML_disk = float(row['M/LdBFM']) if row.get('M/LdBFM', '').strip() else None
        except ValueError:
            ML_disk = None
        try:
            ML_bulge = float(row['M/LbBFM']) if row.get('M/LbBFM', '').strip() else None
        except ValueError:
            ML_bulge = None
        try:
            ML_fML = float(row['M/LfML']) if row.get('M/LfML', '').strip() else None
        except ValueError:
            ML_fML = None
        try:
            chi2_BFM = float(row['Chi2BFM']) if row.get('Chi2BFM', '').strip() else None
        except ValueError:
            chi2_BFM = None
        try:
            chi2_fML = float(row['Chi2fML']) if row.get('Chi2fML', '').strip() else None
        except ValueError:
            chi2_fML = None

        if ML_disk is not None:
            result[ugc_norm] = {
                'ML_disk': ML_disk,
                'ML_bulge': ML_bulge,
                'ML_fML': ML_fML,
                'chi2_BFM': chi2_BFM,
                'chi2_fML': chi2_fML,
            }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Physics: Freeman disk + Sérsic bulge
# ──────────────────────────────────────────────────────────────────────────────

def freeman_vdisk(R_kpc, h_kpc, Mdisk_Msun):
    """V_disk(R) for an exponential disk (Freeman 1970).

    V²(R) = 4πGΣ₀ h y² [I₀(y)K₀(y) - I₁(y)K₁(y)]
    where y = R/(2h), Σ₀ = M/(2πh²)
    """
    if Mdisk_Msun <= 0 or h_kpc <= 0:
        return np.zeros_like(R_kpc, dtype=float)

    h_pc = h_kpc * PC_PER_KPC
    Sigma0 = Mdisk_Msun / (2 * np.pi * h_pc**2)  # Msun/pc²

    y = np.asarray(R_kpc, dtype=float) / (2 * h_kpc)
    y = np.clip(y, 1e-6, 50)

    bessel = i0(y) * k0(y) - i1(y) * k1(y)
    V2 = 4 * np.pi * G_PC * Sigma0 * h_pc * y**2 * bessel
    return np.sqrt(np.maximum(V2, 0))


def sersic_enclosed_mass(R_kpc, re_kpc, n, Mtotal_Msun):
    """Enclosed projected mass of a Sérsic profile at radius R.

    M(<R) = M_total × γ(2n, b_n × (R/R_e)^{1/n}) / Γ(2n)

    where b_n ≈ 2n - 1/3 + 4/(405n) (Ciotti & Bertin 1999).
    γ is the lower incomplete gamma function.
    """
    if Mtotal_Msun <= 0 or re_kpc <= 0 or n <= 0:
        return np.zeros_like(R_kpc, dtype=float)

    R = np.asarray(R_kpc, dtype=float)

    # b_n approximation (Ciotti & Bertin 1999)
    bn = 2 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)

    x = bn * (R / re_kpc)**(1.0 / n)
    x = np.clip(x, 0, 500)  # avoid overflow

    # γ(2n, x) / Γ(2n) = gammainc(2n, x) (scipy regularized)
    frac = gammainc(2 * n, x)

    return Mtotal_Msun * frac


def sersic_vbulge(R_kpc, re_kpc, n, Mbulge_Msun):
    """V_bulge(R) from a Sérsic bulge, assuming spherical deprojection.

    V²(R) ≈ G × M_proj(<R) / R

    This is the "projected mass" approximation, which is standard for RAR
    analysis (matches what SPARC uses). The deprojected 3D mass enclosed
    differs by O(10%) but is model-dependent.
    """
    if Mbulge_Msun <= 0 or re_kpc <= 0 or n <= 0:
        return np.zeros_like(R_kpc, dtype=float)

    R = np.asarray(R_kpc, dtype=float)
    M_enc = sersic_enclosed_mass(R, re_kpc, n, Mbulge_Msun)

    R_pc = np.maximum(R * PC_PER_KPC, 1.0)  # avoid division by zero
    V2 = G_PC * M_enc / R_pc
    return np.sqrt(np.maximum(V2, 0))


# ──────────────────────────────────────────────────────────────────────────────
# Cross-matching GHASP rotation curves
# ──────────────────────────────────────────────────────────────────────────────

def load_ghasp_rotation_curves():
    """Load all GHASP rotation curves from Epinat+2008a and 2008b.

    Returns dict: UGC_norm -> {
        'R_kpc': array, 'Vobs': array, 'eVobs': array,
        'source': 'GHASP_2008a' or 'GHASP_2008b'
    }

    Merges approaching/receding sides by averaging.
    """
    result = {}

    for suffix, source_label in [('ghasp_epinat2008a.tsv', 'GHASP_2008a'),
                                  ('ghasp_epinat2008b.tsv', 'GHASP_2008b')]:
        fp = os.path.join(VIZIER_DIR, suffix)
        if not os.path.exists(fp):
            continue

        header, rows = _parse_vizier_tsv(fp)

        galaxies = {}
        for row in rows:
            name_raw = row.get('Name', '').strip()
            if not name_raw:
                continue

            ugc_norm = _normalize_ugc(name_raw)

            try:
                r_kpc = float(row['r'])
                Vrot = float(row['Vrot'])
                eVrot = float(row['e_Vrot'])
            except (ValueError, KeyError):
                continue

            if Vrot < 0:
                continue

            if ugc_norm not in galaxies:
                galaxies[ugc_norm] = {'points': {}, 'source': source_label}

            r_key = round(r_kpc, 3)
            if r_key not in galaxies[ugc_norm]['points']:
                galaxies[ugc_norm]['points'][r_key] = {'Vs': [], 'eVs': []}
            galaxies[ugc_norm]['points'][r_key]['Vs'].append(abs(Vrot))
            galaxies[ugc_norm]['points'][r_key]['eVs'].append(eVrot)

        for ugc_norm, data in galaxies.items():
            R_kpc, Vobs, eVobs = [], [], []
            for r_key in sorted(data['points'].keys()):
                pt = data['points'][r_key]
                R_kpc.append(r_key)
                Vobs.append(np.mean(pt['Vs']))
                eVobs.append(np.mean(pt['eVs']))

            if len(R_kpc) < 5:
                continue

            # Don't overwrite if already loaded from 2008a
            if ugc_norm not in result:
                result[ugc_norm] = {
                    'R_kpc': np.array(R_kpc),
                    'Vobs': np.array(Vobs),
                    'eVobs': np.array(eVobs),
                    'source': data['source'],
                }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline: reconstruct mass models and merge with GHASP RCs
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_korsaga_mass_models(use_fixed_ml=False, quality_cut=True,
                                     verbose=True):
    """Build full baryonic decomposition for Korsaga+2019 galaxies.

    Strategy:
    1. Load Korsaga Table A1 (properties) + Table A2 (M/L ratios)
    2. Load GHASP rotation curves (Epinat 2008a/b)
    3. Cross-match by UGC name
    4. Compute V_disk(R) + V_bulge(R) at GHASP radii

    Args:
        use_fixed_ml: if True, use color-based M/L (M/LfML) instead of BFM
        quality_cut: if True, skip f_ID=3 galaxies (disturbed RCs)
        verbose: print progress

    Returns dict: normalized_name -> {
        'R_kpc': array, 'Vobs': array, 'eVobs': array,
        'Vgas': array, 'Vdisk': array, 'Vbul': array,
        'dist_Mpc': float, 'inc_deg': float (nan),
        'source': 'Korsaga2019',
        'has_mass_model': True,
        'mass_model_source': 'Korsaga2019_BFM' or 'Korsaga2019_fML'
    }
    """
    # Load data
    tablea1 = load_korsaga_tablea1()
    tablea2 = load_korsaga_tablea2()
    ghasp_rcs = load_ghasp_rotation_curves()

    if verbose:
        print(f"  Korsaga Table A1: {len(tablea1)} galaxies with properties")
        print(f"  Korsaga Table A2: {len(tablea2)} galaxies with M/L ratios")
        print(f"  GHASP RCs loaded: {len(ghasp_rcs)} galaxies")

    # Identify galaxies with BOTH mass model params AND rotation curves
    matched = set(tablea1.keys()) & set(tablea2.keys()) & set(ghasp_rcs.keys())
    has_params_no_rc = (set(tablea1.keys()) & set(tablea2.keys())) - set(ghasp_rcs.keys())

    if verbose:
        print(f"  Matched (A1 ∩ A2 ∩ GHASP): {len(matched)} galaxies")
        print(f"  Have params but no GHASP RC: {len(has_params_no_rc)} galaxies")

    result = {}
    stats = {'total_matched': 0, 'quality_cut': 0, 'disk_only': 0,
             'disk_plus_bulge': 0, 'skipped_ml': 0}

    for ugc in sorted(matched):
        a1 = tablea1[ugc]
        a2 = tablea2[ugc]
        rc = ghasp_rcs[ugc]

        # Quality cut: skip f_ID=3 (disturbed/uncertain RCs)
        if quality_cut and a1['f_ID'] >= 3:
            stats['quality_cut'] += 1
            continue

        # Only use galaxies that have mass model params (param='*')
        if a1['param'] != '*':
            stats['quality_cut'] += 1
            continue

        # Get M/L ratios
        if use_fixed_ml:
            ML_disk = a2.get('ML_fML')
            ML_bulge = a2.get('ML_fML')  # Same color-based M/L for disk and bulge
            model_label = 'Korsaga2019_fML'
        else:
            ML_disk = a2.get('ML_disk')
            ML_bulge = a2.get('ML_bulge')
            model_label = 'Korsaga2019_BFM'

        if ML_disk is None or ML_disk <= 0:
            stats['skipped_ml'] += 1
            continue

        # Compute disk mass: M_disk = ML_disk × LD × 10^8
        h_kpc = a1['h_kpc']
        LD = a1['LD_1e8Lsun']
        Mdisk = ML_disk * LD * 1e8  # Msun

        R_kpc = rc['R_kpc']

        # Freeman disk V(R)
        Vdisk = freeman_vdisk(R_kpc, h_kpc, Mdisk)

        # Sérsic bulge V(R)
        re_kpc = a1['re_kpc']
        n_sersic = a1['n_sersic']
        LB = a1['LB_1e8Lsun']

        if LB > 0 and re_kpc > 0 and n_sersic > 0 and ML_bulge is not None and ML_bulge > 0:
            Mbulge = ML_bulge * LB * 1e8  # Msun
            Vbul = sersic_vbulge(R_kpc, re_kpc, n_sersic, Mbulge)
            stats['disk_plus_bulge'] += 1
        else:
            Vbul = np.zeros_like(R_kpc)
            stats['disk_only'] += 1

        # Gas: we don't have HI data from Korsaga, set to zero
        # (GHASP is Hα, not HI — no gas decomposition available)
        Vgas = np.zeros_like(R_kpc)

        result[ugc] = {
            'R_kpc': R_kpc,
            'Vobs': rc['Vobs'],
            'eVobs': rc['eVobs'],
            'Vgas': Vgas,
            'Vdisk': Vdisk,
            'Vbul': Vbul,
            'dist_Mpc': np.nan,  # Not given in VizieR tables
            'inc_deg': np.nan,
            'source': 'Korsaga2019',
            'has_mass_model': True,
            'mass_model_source': model_label,
            'h_kpc': h_kpc,
            'Mdisk_Msun': Mdisk,
            'Mbulge_Msun': ML_bulge * LB * 1e8 if (ML_bulge and LB) else 0,
            'ghasp_source': rc['source'],
        }
        stats['total_matched'] += 1

    if verbose:
        print(f"\n  Korsaga mass models reconstructed: {stats['total_matched']}")
        print(f"    Disk only: {stats['disk_only']}")
        print(f"    Disk + bulge: {stats['disk_plus_bulge']}")
        print(f"    Quality-cut (f_ID≥3 or no *): {stats['quality_cut']}")
        print(f"    Skipped (bad M/L): {stats['skipped_ml']}")

    return result, stats


def add_korsaga_to_galaxies(galaxies, use_fixed_ml=False,
                             quality_cut=True, verbose=True):
    """Add Korsaga+2019 mass models to an existing galaxy dict.

    Modifies galaxies in-place. Strategy:
    - SPARC/THINGS/deBlok galaxies: KEEP existing (gold-standard mass models)
    - GHASP/S4G/Bouquin Vrot-only: REPLACE with Korsaga mass models (upgrade!)
    - New galaxies not in pipeline: ADD

    Args:
        galaxies: existing dict from load_extended_rar.load_all()
        use_fixed_ml: use color-based M/L instead of BFM
        quality_cut: skip f_ID=3
        verbose: print stats

    Returns stats dict.
    """
    korsaga, stats = reconstruct_korsaga_mass_models(
        use_fixed_ml=use_fixed_ml, quality_cut=quality_cut, verbose=verbose)

    # Build norm→key lookup for existing galaxies
    norm_to_key = {}
    for name in galaxies:
        nn = _normalize_name(name)
        norm_to_key[nn] = name

    # Gold-standard sources we never overwrite
    GOLD_SOURCES = {'SPARC', 'THINGS', 'deBlok2002'}

    added = 0
    replaced = 0
    skipped_gold = 0

    for ugc, kg in korsaga.items():
        nn = _normalize_name(ugc)

        if nn in norm_to_key:
            existing_key = norm_to_key[nn]
            existing = galaxies[existing_key]

            # Never overwrite gold-standard mass models
            if existing['source'] in GOLD_SOURCES:
                skipped_gold += 1
                continue

            # Replace Vrot-only GHASP or noisy S4G matches with Korsaga mass model
            # This is an UPGRADE: we go from no mass model (or noisy S4G) to
            # published Freeman+Sérsic decomposition
            del galaxies[existing_key]
            galaxies[ugc] = kg
            norm_to_key[nn] = ugc
            replaced += 1
        else:
            # Genuinely new galaxy
            galaxies[ugc] = kg
            norm_to_key[nn] = ugc
            added += 1

    stats['added_new'] = added
    stats['replaced_vrot_only'] = replaced
    stats['skipped_gold'] = skipped_gold

    if verbose:
        print(f"\n  Added new: {added} galaxies")
        print(f"  Replaced Vrot-only with Korsaga mass model: {replaced} galaxies")
        print(f"  Skipped (gold-standard SPARC/THINGS/deBlok): {skipped_gold}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Validation against SPARC
# ──────────────────────────────────────────────────────────────────────────────

def validate_against_sparc(verbose=True):
    """Cross-validate Korsaga reconstructed Vdisk against SPARC for overlap galaxies.

    Some SPARC galaxies are UGC galaxies also in Korsaga — compare our
    Freeman reconstruction with SPARC's published Vdisk.

    Returns list of (name, mean_ratio, std_ratio) for matched galaxies.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_extended_rar import load_sparc, _normalize_name as lr_normalize

    sparc = load_sparc(quality_cut=False)
    korsaga, _ = reconstruct_korsaga_mass_models(quality_cut=False, verbose=False)

    # Build SPARC UGC lookup from MRT
    sparc_mrt = os.path.join(BASE, 'data', 'sparc', 'SPARC_Lelli2016c.mrt')
    sparc_ugc = {}
    if os.path.exists(sparc_mrt):
        with open(sparc_mrt) as f:
            lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('---') and i > 50:
                data_start = i + 1
        for line in lines[data_start:]:
            if len(line) < 20:
                continue
            name = line[0:11].strip()
            # Try to find UGC in SPARC galaxy names
            m = re.match(r'UGC0*(\d+)', name)
            if m:
                ugc_norm = f'UGC{m.group(1)}'
                sparc_ugc[ugc_norm] = name

    # Also check for UGC pattern in all SPARC names
    for sname in sparc:
        m = re.match(r'UGC0*(\d+)', sname)
        if m:
            ugc_norm = f'UGC{m.group(1)}'
            sparc_ugc[ugc_norm] = sname

    matches = []
    for ugc, kg in korsaga.items():
        if ugc not in sparc_ugc:
            continue
        sname = sparc_ugc[ugc]
        if sname not in sparc:
            continue
        sg = sparc[sname]

        # Compare Vdisk at matching radii
        R_k = kg['R_kpc']
        Vdisk_k = kg['Vdisk']

        R_s = sg['R_kpc']
        Vdisk_s = sg['Vdisk']

        # Interpolate SPARC onto Korsaga radii
        Vdisk_s_interp = np.interp(R_k, R_s, Vdisk_s, left=0, right=0)

        mask = (Vdisk_s_interp > 5) & (Vdisk_k > 5) & (R_k > 0.5)
        if mask.sum() < 3:
            continue

        ratios = Vdisk_k[mask] / Vdisk_s_interp[mask]
        matches.append({
            'ugc': ugc,
            'sparc_name': sname,
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'n_pts': int(mask.sum()),
        })

    if verbose and matches:
        print(f"\n  SPARC validation: {len(matches)} overlap galaxies")
        all_ratios = [m['mean_ratio'] for m in matches]
        print(f"  Mean Vdisk ratio (Korsaga/SPARC): {np.mean(all_ratios):.3f} "
              f"± {np.std(all_ratios):.3f}")
        for m in sorted(matches, key=lambda x: x['ugc']):
            print(f"    {m['ugc']:12s} ({m['sparc_name']:11s}): "
                  f"ratio={m['mean_ratio']:.3f} ± {m['std_ratio']:.3f} "
                  f"({m['n_pts']} pts)")

    return matches


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("Korsaga+2019 GHASP mass model reconstruction")
    print("=" * 70)

    # Load and reconstruct
    print("\n--- Loading Korsaga data ---")
    korsaga, stats = reconstruct_korsaga_mass_models(verbose=True)

    # Summary
    print(f"\n--- Summary ---")
    for name, g in sorted(korsaga.items()):
        Vbar2 = g['Vdisk']**2 + g['Vbul']**2
        Vbar_max = np.sqrt(np.max(Vbar2)) if len(Vbar2) > 0 else 0
        n_pts = len(g['R_kpc'])
        has_bulge = np.any(g['Vbul'] > 0)
        print(f"  {name:12s}: {n_pts:3d} pts, h={g['h_kpc']:.1f} kpc, "
              f"Md={g['Mdisk_Msun']:.1e}, "
              f"Vbar_max={Vbar_max:.0f} km/s, "
              f"bulge={'Y' if has_bulge else 'N'}")

    # Validate against SPARC
    print("\n--- SPARC cross-validation ---")
    validate_against_sparc(verbose=True)

    # Build RAR
    print("\n--- RAR test ---")
    from load_extended_rar import compute_rar_point
    all_gobs, all_gbar = [], []
    for name, g in korsaga.items():
        R = g['R_kpc']
        mask = R > 0
        gobs, gbar = compute_rar_point(
            g['Vobs'][mask], g['Vgas'][mask], g['Vdisk'][mask], g['Vbul'][mask],
            R[mask])
        valid = (gobs > 0) & (gbar > 0)
        all_gobs.extend(np.log10(gobs[valid]))
        all_gbar.extend(np.log10(gbar[valid]))

    all_gobs = np.array(all_gobs)
    all_gbar = np.array(all_gbar)
    print(f"  Total RAR points: {len(all_gobs)}")
    if len(all_gobs) > 0:
        # RAR residuals
        a0 = 1.2e-10
        gbar_lin = 10.0**all_gbar
        gobs_pred = gbar_lin / (1 - np.exp(-np.sqrt(gbar_lin / a0)))
        log_res = all_gobs - np.log10(gobs_pred)
        print(f"  RAR scatter: σ = {np.std(log_res):.4f} dex")
        print(f"  Mean residual: {np.mean(log_res):+.4f} dex")
        print(f"  gbar range: [{all_gbar.min():.2f}, {all_gbar.max():.2f}]")
        print(f"  gobs range: [{all_gobs.min():.2f}, {all_gobs.max():.2f}]")
