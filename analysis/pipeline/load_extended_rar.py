#!/usr/bin/env python3
"""
load_extended_rar.py — Unified loader for extended rotation curve datasets.

Builds an extended RAR dataset from multiple sources:
  - SPARC: 175 galaxies (standard, from Lelli+2016) — full baryonic decomposition
  - THINGS (de Blok+2008): 6 new galaxies + 13 cross-validation — mass models
  - LITTLE THINGS (Oh+2015): 9 galaxies with total Vrot only
  - de Blok+Bosma 2002 (VizieR J/A+A/385/816): 24 LSB galaxies — Vgas+Vdisk decomposition
  - GHASP Epinat+2008a (J/MNRAS/388/500): 93 galaxies — Hα Fabry-Perot RCs
  - GHASP Epinat+2008b (J/MNRAS/390/466): 82 galaxies — Hα Fabry-Perot RCs
  - Gomez-Lopez+2019 (J/A+A/631/A71): 110 HRS galaxies — Hα Fabry-Perot RCs
  - Verheijen+2001 (J/A+A/370/765): 41 UMa cluster galaxies — HI RCs
  - PHANGS Lang+2020 (J/ApJ/897/122): 70 galaxies — CO rotation curves

For RAR analysis, we need gbar = Vbar^2/R at each radius, which requires baryonic
decomposition (Vgas, Vdisk, Vbulge). Only SPARC, THINGS mass models, and
de Blok+Bosma 2002 provide this. Other catalogs have total Vrot only.

Usage:
    from load_extended_rar import load_sparc, load_things, load_all, load_vizier_all
"""

import os
import re
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPARC_DIR = os.path.join(BASE, 'data', 'sparc')
THINGS_DIR = os.path.join(BASE, 'data', 'things')
LT_DIR = os.path.join(BASE, 'data', 'little_things')
VIZIER_DIR = os.path.join(BASE, 'data', 'vizier_catalogs')

# Standard quality cuts
Q_MAX = 2
INC_MIN = 30.0
INC_MAX = 85.0
NPTS_MIN = 5

# THINGS galaxies NOT in SPARC (genuinely new)
THINGS_NEW = {'NGC3031', 'NGC3621', 'NGC3627', 'NGC4736', 'NGC4826', 'NGC925'}

# THINGS galaxy distances (Mpc) from de Blok+2008 Table 1
THINGS_DISTANCES = {
    'DDO154': 4.3, 'IC2574': 4.0, 'NGC2366': 3.4, 'NGC2403': 3.2,
    'NGC2841': 14.1, 'NGC2903': 8.9, 'NGC2976': 3.6, 'NGC3031': 3.6,
    'NGC3198': 13.8, 'NGC3521': 10.7, 'NGC3621': 6.6, 'NGC3627': 9.3,
    'NGC4736': 4.7, 'NGC4826': 7.5, 'NGC5055': 10.1, 'NGC6946': 5.9,
    'NGC7331': 14.7, 'NGC7793': 3.9, 'NGC925': 9.2,
}

# THINGS galaxy inclinations (deg) from de Blok+2008
THINGS_INCLINATIONS = {
    'DDO154': 66, 'IC2574': 53, 'NGC2366': 64, 'NGC2403': 63,
    'NGC2841': 74, 'NGC2903': 65, 'NGC2976': 65, 'NGC3031': 59,
    'NGC3198': 72, 'NGC3521': 73, 'NGC3621': 65, 'NGC3627': 62,
    'NGC4736': 41, 'NGC4826': 65, 'NGC5055': 59, 'NGC6946': 33,
    'NGC7331': 76, 'NGC7793': 50, 'NGC925': 66,
}


def _normalize_name(name):
    """Normalize galaxy names for cross-matching."""
    name = name.strip().strip('"')
    name = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
                  r'\1', name, flags=re.IGNORECASE)
    name = re.sub(r'^N(\d)', r'NGC\1', name)
    name = re.sub(r'^U(\d)', r'UGC\1', name)
    return name.upper()


def _parse_vizier_tsv(filepath):
    """Generic VizieR TSV parser. Returns (header, rows) where rows are list of dicts."""
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
            row = {}
            for i, h in enumerate(header):
                if i < len(parts):
                    row[h] = parts[i].strip().strip('"')
                else:
                    row[h] = ''
            rows.append(row)
    return header, rows


def _get_sparc_names():
    """Get set of normalized SPARC galaxy names."""
    names = set()
    table2 = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
    if os.path.exists(table2):
        with open(table2) as f:
            for line in f:
                if len(line) > 11 and not line.startswith('#'):
                    n = line[0:11].strip()
                    if n:
                        names.add(_normalize_name(n))
    return names


def load_sparc(quality_cut=True):
    """Load SPARC rotation curves from table2_rotmods.dat.

    Returns dict: galaxy_name -> {
        'R_kpc': array, 'Vobs': array, 'eVobs': array,
        'Vgas': array, 'Vdisk': array, 'Vbul': array,
        'dist_Mpc': float, 'inc_deg': float, 'quality': int,
        'source': 'SPARC'
    }
    """
    table2 = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')

    # First load galaxy properties from MRT
    props = {}
    mrt = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')
    if os.path.exists(mrt):
        with open(mrt) as f:
            lines = f.readlines()
        # Data starts after last "---" line
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('---') and i > 50:
                data_start = i + 1
        for line in lines[data_start:]:
            if len(line) < 20:
                continue
            name = line[0:11].strip()
            parts = line[11:].split()
            if len(parts) >= 17:
                try:
                    dist = float(parts[1])
                    inc = float(parts[4])
                    qual = int(parts[16])
                    props[name] = {'dist_Mpc': dist, 'inc_deg': inc, 'quality': qual}
                except (ValueError, IndexError):
                    pass

    galaxies = {}
    with open(table2) as f:
        for line in f:
            if len(line.strip()) < 50 or line.startswith('#'):
                continue
            try:
                name = line[0:11].strip()
                if not name:
                    continue
                # Fixed-width format: Dist(12:18) R(19:25) Vobs(26:32) eVobs(33:38) Vgas(39:45) Vdisk(46:52) Vbul(53:59)
                R = float(line[19:25].strip())
                Vobs = float(line[26:32].strip())
                eVobs = float(line[33:38].strip())
                Vgas = float(line[39:45].strip())
                Vdisk = float(line[46:52].strip())
                Vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue

            if name not in galaxies:
                p = props.get(name, {})
                galaxies[name] = {
                    'R_kpc': [], 'Vobs': [], 'eVobs': [],
                    'Vgas': [], 'Vdisk': [], 'Vbul': [],
                    'dist_Mpc': p.get('dist_Mpc', np.nan),
                    'inc_deg': p.get('inc_deg', np.nan),
                    'quality': p.get('quality', 9),
                    'source': 'SPARC',
                }
            galaxies[name]['R_kpc'].append(R)
            galaxies[name]['Vobs'].append(Vobs)
            galaxies[name]['eVobs'].append(eVobs)
            galaxies[name]['Vgas'].append(Vgas)
            galaxies[name]['Vdisk'].append(Vdisk)
            galaxies[name]['Vbul'].append(Vbul)

    # Convert to arrays and apply quality cuts
    result = {}
    for name, g in galaxies.items():
        for key in ['R_kpc', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
            g[key] = np.array(g[key])

        if quality_cut:
            if g['quality'] > Q_MAX:
                continue
            if not (INC_MIN < g['inc_deg'] < INC_MAX):
                continue
            if len(g['R_kpc']) < NPTS_MIN:
                continue

        result[name] = g

    return result


def load_things(only_new=False, model='ISO.free.REV'):
    """Load THINGS rotation curves + mass models from de Blok+2008.

    Args:
        only_new: if True, only return galaxies NOT in SPARC
        model: mass model variant (default: ISO.free.REV = pseudo-isothermal, free fit)

    Returns dict: galaxy_name -> {
        'R_kpc': array, 'Vobs': array, 'eVobs': array,
        'Vgas': array, 'Vdisk': array, 'Vbul': array,
        'R_arcsec': array, 'inc_deg': float,
        'dist_Mpc': float, 'source': 'THINGS',
        'has_mass_model': bool
    }
    """
    curves_dir = os.path.join(THINGS_DIR, 'curves')
    models_dir = os.path.join(THINGS_DIR, 'mass_models')
    result = {}

    if not os.path.isdir(curves_dir):
        return result

    for fn in sorted(os.listdir(curves_dir)):
        if not fn.endswith('.curve.02'):
            continue
        name = fn.replace('.curve.02', '')

        if only_new and name not in THINGS_NEW:
            continue

        # Apply inclination cut
        inc = THINGS_INCLINATIONS.get(name, 0)
        if not (INC_MIN < inc < INC_MAX):
            continue

        # Load rotation curve
        R_arcsec, Vobs, eVobs = [], [], []
        with open(os.path.join(curves_dir, fn)) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    r = float(parts[0])
                    v = float(parts[1])
                    err = float(parts[6])
                except (ValueError, IndexError):
                    continue
                R_arcsec.append(r)
                Vobs.append(v)
                eVobs.append(err)

        if len(R_arcsec) < NPTS_MIN:
            continue

        R_arcsec = np.array(R_arcsec)
        Vobs = np.array(Vobs)
        eVobs = np.array(eVobs)

        # Convert arcsec to kpc
        dist = THINGS_DISTANCES.get(name, np.nan)
        R_kpc = R_arcsec * dist / 206.265

        entry = {
            'R_kpc': R_kpc, 'R_arcsec': R_arcsec,
            'Vobs': Vobs, 'eVobs': eVobs,
            'dist_Mpc': dist,
            'inc_deg': inc,
            'source': 'THINGS',
            'has_mass_model': False,
            'Vgas': np.zeros_like(R_kpc),
            'Vdisk': np.zeros_like(R_kpc),
            'Vbul': np.zeros_like(R_kpc),
        }

        # Try to load mass model
        model_file = os.path.join(models_dir, f'{name}.{model}.dat')
        if os.path.exists(model_file) and os.path.getsize(model_file) > 100:
            try:
                R_mod, Vgas_mod, Vdisk_mod, Vbul_mod, Vobs_mod, eVobs_mod = \
                    _parse_things_mass_model(model_file)

                entry['R_kpc'] = R_mod
                entry['Vobs'] = Vobs_mod
                entry['eVobs'] = eVobs_mod
                entry['Vgas'] = Vgas_mod
                entry['Vdisk'] = Vdisk_mod
                entry['Vbul'] = Vbul_mod
                entry['has_mass_model'] = True
            except Exception:
                pass

        result[name] = entry

    return result


def _parse_things_mass_model(filepath):
    """Parse THINGS ROTMAS mass model file.

    Format: R(kpc) | Vgas | Vdisk | Vbulge | Vobs | eVobs | Vhalo | Vtotal | ...
    """
    R, Vgas, Vdisk, Vbul, Vobs, eVobs = [], [], [], [], [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith('!') or line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                r = float(parts[0])
                vg = float(parts[1])
                vd = float(parts[2])
                vb = float(parts[3])
                vo = float(parts[4])
                ev = float(parts[5])
            except ValueError:
                continue
            R.append(r)
            Vgas.append(vg)
            Vdisk.append(vd)
            Vbul.append(vb)
            Vobs.append(vo)
            eVobs.append(ev)

    return (np.array(R), np.array(Vgas), np.array(Vdisk),
            np.array(Vbul), np.array(Vobs), np.array(eVobs))


def load_little_things():
    """Load LITTLE THINGS rotation curves from Oh+2015 Figure 2 data.

    Only 9 galaxies have machine-readable rotation curves (total Vrot only,
    NO baryonic decomposition). These are useful for gobs but not gbar.
    """
    lt_props = {
        'CVnIdwA':  {'dist': 3.6, 'inc': 66.5},
        'DDO_210':  {'dist': 0.9, 'inc': 66.7},
        'DDO_216':  {'dist': 1.1, 'inc': 63.7},
        'DDO_53':   {'dist': 3.6, 'inc': 27.0},
        'DDO_70':   {'dist': 1.3, 'inc': 50.0},
        'IC_10':    {'dist': 0.7, 'inc': 47.0},
        'IC_1613':  {'dist': 0.7, 'inc': 48.0},
        'UGC_8508': {'dist': 2.6, 'inc': 82.5},
        'WLM':      {'dist': 1.0, 'inc': 74.0},
    }

    data_file = os.path.join(LT_DIR, 'aj513259f2', 'dbf2A.txt')
    if not os.path.exists(data_file):
        return {}

    raw = {}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in '#!=\x00':
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            name = parts[0]
            dtype = parts[1]
            if dtype != 'Data':
                continue
            try:
                R03 = float(parts[2])
                V03 = float(parts[3])
                R_scaled = float(parts[4])
                V_scaled = float(parts[5])
                eV_scaled = float(parts[6])
            except ValueError:
                continue

            if name not in raw:
                raw[name] = {'R03': R03, 'V03': V03, 'points': []}
            raw[name]['points'].append((R_scaled * R03, V_scaled * V03, eV_scaled * V03))

    result = {}
    for name, data in raw.items():
        p = lt_props.get(name, {})
        inc = p.get('inc', 0)

        if not (INC_MIN < inc < INC_MAX):
            continue

        pts = np.array(data['points'])
        if len(pts) < NPTS_MIN:
            continue

        result[name] = {
            'R_kpc': pts[:, 0],
            'Vobs': pts[:, 1],
            'eVobs': pts[:, 2],
            'Vgas': np.zeros(len(pts)),
            'Vdisk': np.zeros(len(pts)),
            'Vbul': np.zeros(len(pts)),
            'dist_Mpc': p.get('dist', np.nan),
            'inc_deg': inc,
            'source': 'LITTLE_THINGS',
            'has_mass_model': False,
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# VizieR catalog loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_deblok2002():
    """Load de Blok & Bosma 2002 LSB rotation curves with baryonic decomposition.

    Source: VizieR J/A+A/385/816 (processed table with Vgas, Vdisk)
    24 LSB galaxies, 484 data points, WITH Vgas+Vdisk decomposition (no bulge).
    Some galaxies lack Vgas or Vdisk (empty fields) — those still get loaded
    but flagged with partial decomposition.

    Returns dict: galaxy_name -> standard galaxy dict
    """
    fp = os.path.join(VIZIER_DIR, 'deblok2002_processed.tsv')
    if not os.path.exists(fp):
        return {}

    header, rows = _parse_vizier_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        try:
            R_kpc = float(row['r(kpc)'])
            Vrot = float(row['Vrot'])
            eVrot = float(row['e_Vrot'])
        except (ValueError, KeyError):
            continue

        # Vgas and Vdisk may be empty for some galaxies
        try:
            Vgas = float(row['Vgas']) if row.get('Vgas', '') else 0.0
        except ValueError:
            Vgas = 0.0
        try:
            Vdisk = float(row['Vdisk']) if row.get('Vdisk', '') else 0.0
        except ValueError:
            Vdisk = 0.0

        if name not in galaxies:
            has_vgas = row.get('Vgas', '') != ''
            has_vdisk = row.get('Vdisk', '') != ''
            galaxies[name] = {
                'R_kpc': [], 'Vobs': [], 'eVobs': [],
                'Vgas': [], 'Vdisk': [], 'Vbul': [],
                'source': 'deBlok2002',
                'has_mass_model': has_vgas or has_vdisk,
                'dist_Mpc': np.nan,
                'inc_deg': np.nan,
            }

        galaxies[name]['R_kpc'].append(R_kpc)
        galaxies[name]['Vobs'].append(Vrot)
        galaxies[name]['eVobs'].append(eVrot)
        galaxies[name]['Vgas'].append(Vgas)
        galaxies[name]['Vdisk'].append(Vdisk)
        galaxies[name]['Vbul'].append(0.0)  # LSB galaxies, no bulge

    result = {}
    for name, g in galaxies.items():
        for key in ['R_kpc', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
            g[key] = np.array(g[key])
        if len(g['R_kpc']) >= NPTS_MIN:
            result[name] = g

    return result


def load_ghasp_epinat2008a():
    """Load GHASP Epinat+2008a Hα Fabry-Perot rotation curves.

    Source: VizieR J/MNRAS/388/500
    93 galaxies with approaching and receding side RCs.
    r is in kpc, Vrot in km/s. Both sides ('a' and 'r') available.
    We average approaching/receding sides for each radial bin.

    No baryonic decomposition — total Vrot only.
    """
    fp = os.path.join(VIZIER_DIR, 'ghasp_epinat2008a.tsv')
    if not os.path.exists(fp):
        return {}

    header, rows = _parse_vizier_tsv(fp)

    # Collect all points per galaxy, keyed by (name, radius_bin)
    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        try:
            r_kpc = float(row['r'])
            Vrot = float(row['Vrot'])
            eVrot = float(row['e_Vrot'])
        except (ValueError, KeyError):
            continue

        if Vrot < 0:
            continue  # Some approaching-side values are negative velocities

        if name not in galaxies:
            galaxies[name] = {'points': {}}

        # Key by rounded radius to merge approaching/receding
        r_key = round(r_kpc, 3)
        if r_key not in galaxies[name]['points']:
            galaxies[name]['points'][r_key] = {'Vs': [], 'eVs': []}
        galaxies[name]['points'][r_key]['Vs'].append(abs(Vrot))
        galaxies[name]['points'][r_key]['eVs'].append(eVrot)

    result = {}
    for name, data in galaxies.items():
        R_kpc, Vobs, eVobs = [], [], []
        for r_key in sorted(data['points'].keys()):
            pt = data['points'][r_key]
            R_kpc.append(r_key)
            Vobs.append(np.mean(pt['Vs']))
            eVobs.append(np.mean(pt['eVs']))

        if len(R_kpc) < NPTS_MIN:
            continue

        result[name] = {
            'R_kpc': np.array(R_kpc),
            'Vobs': np.array(Vobs),
            'eVobs': np.array(eVobs),
            'Vgas': np.zeros(len(R_kpc)),
            'Vdisk': np.zeros(len(R_kpc)),
            'Vbul': np.zeros(len(R_kpc)),
            'source': 'GHASP_2008a',
            'has_mass_model': False,
            'dist_Mpc': np.nan,
            'inc_deg': np.nan,
        }

    return result


def load_ghasp_epinat2008b():
    """Load GHASP Epinat+2008b rotation curves.

    Source: VizieR J/MNRAS/390/466
    82 galaxies, same format as 2008a. Average sides.
    """
    fp = os.path.join(VIZIER_DIR, 'ghasp_epinat2008b.tsv')
    if not os.path.exists(fp):
        return {}

    header, rows = _parse_vizier_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        try:
            r_kpc = float(row['r'])
            Vrot = float(row['Vrot'])
            eVrot = float(row['e_Vrot'])
        except (ValueError, KeyError):
            continue

        if Vrot < 0:
            continue

        if name not in galaxies:
            galaxies[name] = {'points': {}}

        r_key = round(r_kpc, 3)
        if r_key not in galaxies[name]['points']:
            galaxies[name]['points'][r_key] = {'Vs': [], 'eVs': []}
        galaxies[name]['points'][r_key]['Vs'].append(abs(Vrot))
        galaxies[name]['points'][r_key]['eVs'].append(eVrot)

    result = {}
    for name, data in galaxies.items():
        R_kpc, Vobs, eVobs = [], [], []
        for r_key in sorted(data['points'].keys()):
            pt = data['points'][r_key]
            R_kpc.append(r_key)
            Vobs.append(np.mean(pt['Vs']))
            eVobs.append(np.mean(pt['eVs']))

        if len(R_kpc) < NPTS_MIN:
            continue

        result[name] = {
            'R_kpc': np.array(R_kpc),
            'Vobs': np.array(Vobs),
            'eVobs': np.array(eVobs),
            'Vgas': np.zeros(len(R_kpc)),
            'Vdisk': np.zeros(len(R_kpc)),
            'Vbul': np.zeros(len(R_kpc)),
            'source': 'GHASP_2008b',
            'has_mass_model': False,
            'dist_Mpc': np.nan,
            'inc_deg': np.nan,
        }

    return result


def _load_hrs_crossmatch():
    """Load HRS number -> galaxy name mapping from Boselli+2010."""
    fp = os.path.join(VIZIER_DIR, 'hrs_boselli2010.tsv')
    if not os.path.exists(fp):
        return {}

    hrs_to_name = {}
    with open(fp) as f:
        for line in f:
            if line.startswith('#') or line.startswith('-') or not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 6 or parts[0].strip() in ('HRS', ''):
                continue
            hrs = parts[0].strip()
            ngc = parts[1].strip()
            m_ngc = parts[2].strip()
            ugc = parts[3].strip()
            ic = parts[4].strip()
            simbad = parts[5].strip()

            if ngc:
                name = f'NGC{ngc}{m_ngc}'
            elif ugc:
                name = f'UGC{ugc}'
            elif ic:
                name = f'IC{ic}'
            else:
                name = simbad
            hrs_to_name[hrs] = name

    return hrs_to_name


def load_gomezlopez2019():
    """Load Gomez-Lopez+2019 GHASP Hα Fabry-Perot rotation curves.

    Source: VizieR J/A+A/631/A71
    ~110 HRS galaxies. Uses HRS number IDs — cross-matched via Boselli+2010.
    r is in kpc (column 'r'), v is in km/s. Both sides available.
    """
    fp = os.path.join(VIZIER_DIR, 'ghasp_gomezlopez2019.tsv')
    if not os.path.exists(fp):
        return {}

    hrs_map = _load_hrs_crossmatch()
    if not hrs_map:
        return {}

    header, rows = _parse_vizier_tsv(fp)

    galaxies = {}
    for row in rows:
        hrs_id = row.get('HRS', '').strip()
        if not hrs_id or hrs_id not in hrs_map:
            continue

        name = hrs_map[hrs_id]

        try:
            r_kpc = float(row['r'])
            Vrot = float(row['v'])
            eVrot = float(row['s_v'])
        except (ValueError, KeyError):
            continue

        if Vrot < 0:
            continue

        if name not in galaxies:
            galaxies[name] = {'points': {}}

        r_key = round(r_kpc, 3)
        if r_key not in galaxies[name]['points']:
            galaxies[name]['points'][r_key] = {'Vs': [], 'eVs': []}
        galaxies[name]['points'][r_key]['Vs'].append(abs(Vrot))
        galaxies[name]['points'][r_key]['eVs'].append(eVrot)

    result = {}
    for name, data in galaxies.items():
        R_kpc, Vobs, eVobs = [], [], []
        for r_key in sorted(data['points'].keys()):
            pt = data['points'][r_key]
            R_kpc.append(r_key)
            Vobs.append(np.mean(pt['Vs']))
            eVobs.append(np.mean(pt['eVs']))

        if len(R_kpc) < NPTS_MIN:
            continue

        result[name] = {
            'R_kpc': np.array(R_kpc),
            'Vobs': np.array(Vobs),
            'eVobs': np.array(eVobs),
            'Vgas': np.zeros(len(R_kpc)),
            'Vdisk': np.zeros(len(R_kpc)),
            'Vbul': np.zeros(len(R_kpc)),
            'source': 'GomezLopez2019',
            'has_mass_model': False,
            'dist_Mpc': np.nan,
            'inc_deg': np.nan,
        }

    return result


def load_verheijen2001():
    """Load Verheijen+2001 UMa cluster HI rotation curves.

    Source: VizieR J/A+A/370/765
    41 UMa galaxies. Rad is in arcsec, Vrot in km/s.
    Distance to UMa = 18.6 Mpc (Verheijen & Sancisi 2001).
    Inclination provided per data point.
    """
    fp = os.path.join(VIZIER_DIR, 'verheijen2001_rc.tsv')
    if not os.path.exists(fp):
        return {}

    UMA_DIST = 18.6  # Mpc

    header, rows = _parse_vizier_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        try:
            R_arcsec = float(row['Rad'])
            Vrot = float(row['Vrot']) if row.get('Vrot', '') else np.nan
            inc = float(row['Incl']) if row.get('Incl', '') else np.nan
        except (ValueError, KeyError):
            continue

        if np.isnan(Vrot) or Vrot <= 0:
            continue

        # Convert arcsec to kpc
        R_kpc = R_arcsec * UMA_DIST / 206.265

        if name not in galaxies:
            galaxies[name] = {
                'R_kpc': [], 'Vobs': [], 'eVobs': [],
                'inc_deg': inc, 'dist_Mpc': UMA_DIST,
            }
        galaxies[name]['R_kpc'].append(R_kpc)
        galaxies[name]['Vobs'].append(Vrot)
        # Error: average of VrotAppM and VrotAppm (asymmetric errors)
        try:
            ehi = float(row.get('VrotAppM', '10'))
            elo = float(row.get('VrotAppm', '10'))
            galaxies[name]['eVobs'].append((ehi + elo) / 2)
        except ValueError:
            galaxies[name]['eVobs'].append(10.0)

    result = {}
    for name, g in galaxies.items():
        n = len(g['R_kpc'])
        if n < NPTS_MIN:
            continue

        # Normalize name (N3726 -> NGC3726)
        norm_name = _normalize_name(name)

        result[norm_name] = {
            'R_kpc': np.array(g['R_kpc']),
            'Vobs': np.array(g['Vobs']),
            'eVobs': np.array(g['eVobs']),
            'Vgas': np.zeros(n),
            'Vdisk': np.zeros(n),
            'Vbul': np.zeros(n),
            'source': 'Verheijen2001',
            'has_mass_model': False,
            'dist_Mpc': g['dist_Mpc'],
            'inc_deg': g['inc_deg'],
        }

    return result


def load_phangs_lang2020():
    """Load PHANGS Lang+2020 CO rotation curves.

    Source: VizieR J/ApJ/897/122
    70 galaxies. Rad is in kpc, VRot in km/s.
    Two error columns: E_VRot (statistical) and e_VRot (systematic).
    """
    fp = os.path.join(VIZIER_DIR, 'phangs_lang2020.tsv')
    if not os.path.exists(fp):
        return {}

    header, rows = _parse_vizier_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('ID', '').strip()
        if not name:
            continue

        try:
            R_kpc = float(row['Rad'])
            Vrot = float(row['VRot'])
        except (ValueError, KeyError):
            continue

        # Combine statistical + systematic error in quadrature
        try:
            e_stat = float(row.get('E_VRot', '0'))
            e_sys = float(row.get('e_VRot', '0'))
            eVrot = np.sqrt(e_stat**2 + e_sys**2)
        except ValueError:
            eVrot = 5.0

        if name not in galaxies:
            galaxies[name] = {'R_kpc': [], 'Vobs': [], 'eVobs': []}
        galaxies[name]['R_kpc'].append(R_kpc)
        galaxies[name]['Vobs'].append(Vrot)
        galaxies[name]['eVobs'].append(eVrot)

    result = {}
    for name, g in galaxies.items():
        n = len(g['R_kpc'])
        if n < NPTS_MIN:
            continue

        norm_name = _normalize_name(name)

        result[norm_name] = {
            'R_kpc': np.array(g['R_kpc']),
            'Vobs': np.array(g['Vobs']),
            'eVobs': np.array(g['eVobs']),
            'Vgas': np.zeros(n),
            'Vdisk': np.zeros(n),
            'Vbul': np.zeros(n),
            'source': 'PHANGS_Lang2020',
            'has_mass_model': False,
            'dist_Mpc': np.nan,
            'inc_deg': np.nan,
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# RAR computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_rar_point(Vobs, Vgas, Vdisk, Vbul, R_kpc, ML_disk=1.0, ML_bul=1.0):
    """Compute gobs and gbar for a single data point.

    gobs = Vobs^2 / R
    gbar = (Vgas|Vgas| + ML_disk*Vdisk|Vdisk| + ML_bul*Vbul|Vbul|) / R

    Returns gobs, gbar in m/s^2.
    """
    kpc_to_m = 3.0857e19
    kms_to_ms = 1e3

    R_m = R_kpc * kpc_to_m
    gobs = (Vobs * kms_to_ms)**2 / R_m

    vbar2 = (np.sign(Vgas) * Vgas**2 +
             ML_disk * np.sign(Vdisk) * Vdisk**2 +
             ML_bul * np.sign(Vbul) * Vbul**2)
    gbar = np.abs(vbar2) * kms_to_ms**2 / R_m

    return gobs, gbar


def build_rar(galaxies, ML_disk=1.0, ML_bul=1.0):
    """Build RAR arrays from a galaxy dict.

    Only includes galaxies with baryonic decomposition (SPARC or has_mass_model).

    Returns:
        log_gobs: array of log10(gobs) in m/s^2
        log_gbar: array of log10(gbar) in m/s^2
        galaxy_names: array of galaxy names (one per point)
        sources: array of source labels
    """
    all_gobs, all_gbar, all_names, all_sources = [], [], [], []

    for name, g in galaxies.items():
        if not g.get('has_mass_model', False) and g['source'] != 'SPARC':
            continue  # Skip galaxies without baryonic decomposition

        R = g['R_kpc']
        mask = R > 0  # Avoid division by zero

        gobs, gbar = compute_rar_point(
            g['Vobs'][mask], g['Vgas'][mask], g['Vdisk'][mask], g['Vbul'][mask],
            R[mask], ML_disk, ML_bul
        )

        valid = (gobs > 0) & (gbar > 0)
        all_gobs.extend(np.log10(gobs[valid]))
        all_gbar.extend(np.log10(gbar[valid]))
        all_names.extend([name] * valid.sum())
        all_sources.extend([g['source']] * valid.sum())

    return (np.array(all_gobs), np.array(all_gbar),
            np.array(all_names), np.array(all_sources))


# ──────────────────────────────────────────────────────────────────────────────
# Combined loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_vizier_all(skip_sparc_overlap=True):
    """Load all VizieR rotation curve catalogs.

    Args:
        skip_sparc_overlap: if True, exclude galaxies that are already in SPARC.

    Returns combined galaxy dict.
    """
    sparc_norm = _get_sparc_names() if skip_sparc_overlap else set()

    loaders = [
        ('deBlok2002', load_deblok2002),
        ('GHASP_2008a', load_ghasp_epinat2008a),
        ('GHASP_2008b', load_ghasp_epinat2008b),
        ('GomezLopez2019', load_gomezlopez2019),
        ('Verheijen2001', load_verheijen2001),
        ('PHANGS_Lang2020', load_phangs_lang2020),
    ]

    combined = {}
    for label, loader_fn in loaders:
        data = loader_fn()
        added = 0
        for name, g in data.items():
            nn = _normalize_name(name)
            if skip_sparc_overlap and nn in sparc_norm:
                continue
            if nn not in combined:
                combined[nn] = g
                added += 1
        print(f"  {label}: {len(data)} galaxies loaded, {added} new (not in SPARC or earlier catalogs)")

    return combined


def load_all(include_sparc_overlap=False, include_vizier=False,
             match_photometry=False):
    """Load all datasets combined.

    Args:
        include_sparc_overlap: if True, include THINGS galaxies that are also in SPARC.
        include_vizier: if True, also load all VizieR catalogs.
        match_photometry: if True, match Vrot-only galaxies with S4G+ALFALFA
                          for baryonic decomposition.

    Returns combined galaxy dict.
    """
    galaxies = load_sparc(quality_cut=True)
    print(f"SPARC: {len(galaxies)} galaxies")

    things = load_things(only_new=not include_sparc_overlap)
    for name, g in things.items():
        if name not in galaxies:
            galaxies[name] = g
    print(f"THINGS new: {sum(1 for g in things.values() if g['source'] == 'THINGS')} galaxies "
          f"({sum(1 for n in things if n in THINGS_NEW)} genuinely new)")

    lt = load_little_things()
    for name, g in lt.items():
        if name not in galaxies:
            galaxies[name] = g
    print(f"LITTLE THINGS: {len(lt)} galaxies (total Vrot only, no baryonic decomposition)")

    if include_vizier:
        print("\nLoading VizieR catalogs...")
        sparc_norm = {_normalize_name(n) for n in galaxies}
        vizier = load_vizier_all(skip_sparc_overlap=False)
        added = 0
        for name, g in vizier.items():
            nn = _normalize_name(name)
            if nn not in sparc_norm:
                galaxies[name] = g
                sparc_norm.add(nn)
                added += 1
        print(f"VizieR total new: {added} galaxies")

    if match_photometry:
        print("\nMatching with S4G + ALFALFA photometry...")
        from match_photometry import compute_baryonic_components
        compute_baryonic_components(galaxies)

    print(f"\nTotal: {len(galaxies)} unique galaxies")

    n_rar = sum(1 for g in galaxies.values()
                if g.get('has_mass_model', False) or g['source'] == 'SPARC')
    print(f"Usable for RAR (with baryonic decomposition): {n_rar}")

    return galaxies


if __name__ == '__main__':
    print("=" * 70)
    print("Extended RAR dataset loader")
    print("=" * 70)

    galaxies = load_all(include_sparc_overlap=True, include_vizier=True,
                        match_photometry=True)

    print("\n--- Dataset breakdown ---")
    by_source = {}
    for name, g in galaxies.items():
        src = g['source']
        by_source.setdefault(src, []).append((name, g))

    for src, gs in sorted(by_source.items()):
        n_pts = sum(len(g['R_kpc']) for _, g in gs)
        n_mm = sum(1 for _, g in gs if g.get('has_mass_model', False) or g['source'] == 'SPARC')
        print(f"  {src}: {len(gs)} galaxies, {n_pts} data points, {n_mm} with mass models")

    print("\n--- THINGS galaxies (new, not in SPARC) ---")
    for name in sorted(THINGS_NEW):
        if name in galaxies:
            g = galaxies[name]
            mm = "with mass model" if g.get('has_mass_model') else "RC only"
            print(f"  {name}: {len(g['R_kpc'])} pts, D={g['dist_Mpc']:.1f} Mpc, "
                  f"inc={g['inc_deg']:.0f}°, {mm}")
        else:
            print(f"  {name}: EXCLUDED (quality/inclination cut)")

    # Build RAR
    print("\n--- RAR construction ---")
    log_gobs, log_gbar, names, sources = build_rar(galaxies)
    print(f"Total RAR points: {len(log_gobs)}")
    if len(log_gobs) > 0:
        print(f"  gobs range: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")
        print(f"  gbar range: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")

        # Points by source
        for src in sorted(set(sources)):
            mask = sources == src
            print(f"  {src}: {mask.sum()} RAR points")

    # Summary of Vrot-only galaxies
    vrot_only = sum(1 for g in galaxies.values()
                    if not g.get('has_mass_model', False) and g['source'] != 'SPARC')
    print(f"\nVrot-only galaxies (need photometry for RAR): {vrot_only}")
