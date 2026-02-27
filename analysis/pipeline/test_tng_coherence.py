#!/usr/bin/env python3
"""
TNG Coherence Analysis — Run LOCALLY on Extracted Profiles
===========================================================

Companion to tng_extract_profiles.py. After downloading tng_mass_profiles.npz
from the TNG JupyterLab, this script computes:
  - Per-galaxy RAR residuals at 15 radial points
  - Lag-1 ACF (demeaned)
  - Lomb-Scargle periodicity with permutation null
  - Comparison to SPARC reference values

Prerequisites:
  Place tng_mass_profiles.npz in ~/Desktop/tng_cross_validation/

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
SPARC_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
TNG_PROFILES = os.path.expanduser('~/Desktop/tng_cross_validation/tng_mass_profiles.npz')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
G = 6.674e-11        # m^3 kg^-1 s^-2
M_sun = 1.989e30     # kg
kpc_m = 3.086e19     # m per kpc
g_dagger = 1.20e-10  # m/s^2
N_SURR = 200
N_BOOT = 10000
MIN_PTS = 10         # Minimum valid radii per galaxy

np.random.seed(42)

print("=" * 76)
print("TNG COHERENCE ANALYSIS — 15-Radii Profiles")
print("=" * 76)


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def lag_autocorrelation(x, lag=1, center=True):
    n = len(x)
    if n <= lag + 1:
        return np.nan
    x_use = x - np.mean(x) if center else x.copy()
    var = np.mean(x_use**2)
    if var < 1e-30:
        return np.nan
    return np.mean(x_use[:n - lag] * x_use[lag:]) / var


def run_ls_test(R, eps, n_surr=N_SURR, rng=None):
    """Lomb-Scargle with permutation null (identical to SPARC pipeline)."""
    n = len(R)
    if n < 5:
        return False, np.nan, np.nan, 1.0

    var_eps = np.var(eps)
    s_param = n * var_eps * 0.5
    try:
        k = min(3, n - 1)
        if k >= 1:
            spline = UnivariateSpline(R, eps, k=k, s=s_param)
            eps_det = eps - spline(R)
        else:
            eps_det = eps - np.mean(eps)
    except Exception:
        eps_det = eps - np.mean(eps)

    std_det = np.std(eps_det)
    if std_det < 1e-30:
        return False, np.nan, np.nan, 1.0

    y = (eps_det - np.mean(eps_det)) / std_det
    R_ext = R[-1] - R[0]
    if R_ext <= 0:
        return False, np.nan, np.nan, 1.0

    f_min = 1.0 / R_ext
    f_max = (n / 2.0) / R_ext
    n_freq = min(500, 10 * n)
    if f_max <= f_min:
        return False, np.nan, np.nan, 1.0

    freq_grid = np.linspace(f_min, f_max, n_freq)
    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    power_peak = float(power[idx_peak])
    wl_kpc = 1.0 / freq_grid[idx_peak]

    if rng is None:
        rng = np.random.default_rng(789)
    null_peaks = np.array([np.max(LombScargle(R, rng.permutation(y),
                           fit_mean=False, center_data=True).power(freq_grid))
                           for _ in range(n_surr)])
    perm_p = max(float(np.mean(null_peaks >= power_peak)), 1.0 / (n_surr + 1))

    return perm_p < 0.05, float(wl_kpc), power_peak, perm_p


# ══════════════════════════════════════════════════════════════════════
#  1. LOAD TNG PROFILES
# ══════════════════════════════════════════════════════════════════════
print(f"\n[1] Loading TNG profiles from {TNG_PROFILES}...")

if not os.path.exists(TNG_PROFILES):
    print(f"\n  ERROR: {TNG_PROFILES} not found!")
    print("  Run tng_extract_profiles.py on the TNG JupyterLab first,")
    print("  then download tng_mass_profiles.npz to ~/Desktop/tng_cross_validation/")
    import sys; sys.exit(1)

data = np.load(TNG_PROFILES, allow_pickle=True)
galaxy_ids = data['galaxy_ids']
r_half = data['r_half_kpc']
vmax = data['vmax']
m_star_total = data['m_star_total']
radii = data['radii_kpc']       # (N, 15)
m_star_enc = data['m_star_enc']  # (N, 15)
m_gas_enc = data['m_gas_enc']
m_dm_enc = data['m_dm_enc']
r_multipliers = data['r_multipliers']

n_total = len(galaxy_ids)
print(f"  Loaded {n_total} galaxies with {radii.shape[1]} radial bins each")


# ══════════════════════════════════════════════════════════════════════
#  2. COMPUTE RAR RESIDUALS PER GALAXY
# ══════════════════════════════════════════════════════════════════════
print(f"\n[2] Computing per-galaxy RAR residuals at 15 radii...")

tng_galaxies = []

for i in range(n_total):
    R_kpc = radii[i]
    ms = m_star_enc[i]
    mg = m_gas_enc[i]
    md = m_dm_enc[i]

    # Baryonic mass = stars + gas; total = stars + gas + DM
    m_bar = ms + mg
    m_tot = m_bar + md

    # Compute accelerations at each radius
    valid = (R_kpc > 0) & (m_bar > 0) & (m_tot > m_bar)
    if np.sum(valid) < MIN_PTS:
        continue

    R_valid = R_kpc[valid]
    r_m = R_valid * kpc_m
    gb = G * m_bar[valid] * M_sun / r_m**2
    go = G * m_tot[valid] * M_sun / r_m**2

    mask2 = (gb > 1e-15) & (go > 1e-15)
    if np.sum(mask2) < MIN_PTS:
        continue

    R_use = R_valid[mask2]
    log_gb = np.log10(gb[mask2])
    log_go = np.log10(go[mask2])

    residuals = log_go - rar_function(log_gb)
    residuals_dm = residuals - np.mean(residuals)

    r1_dm = lag_autocorrelation(residuals_dm, lag=1, center=True)
    if np.isnan(r1_dm):
        continue

    tng_galaxies.append({
        'gid': int(galaxy_ids[i]),
        'n_pts': len(R_use),
        'r1_dm': float(r1_dm),
        'R': R_use,
        'residuals_dm': residuals_dm,
        'r_half': float(r_half[i]),
        'vmax': float(vmax[i]),
        'm_star': float(m_star_total[i]),
    })

n_tng = len(tng_galaxies)
print(f"  TNG galaxies with ≥{MIN_PTS} valid radii: {n_tng}")


# ══════════════════════════════════════════════════════════════════════
#  3. LOMB-SCARGLE FOR TNG GALAXIES
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3] TNG Lomb-Scargle periodograms ({N_SURR} surrogates/galaxy)...")

perm_rng = np.random.default_rng(789)

for gi, g in enumerate(tng_galaxies):
    detected, wl, power, perm_p = run_ls_test(g['R'], g['residuals_dm'], rng=perm_rng)
    g['ls_detected'] = detected
    g['wl_kpc'] = wl
    g['power_peak'] = power
    g['perm_p'] = perm_p

    if (gi + 1) % 500 == 0:
        print(f"    {gi+1}/{n_tng}...")

print(f"  Done.")


# ══════════════════════════════════════════════════════════════════════
#  4. LOAD SPARC REFERENCE (recompute)
# ══════════════════════════════════════════════════════════════════════
print(f"\n[4] Loading SPARC reference...")

table2_path = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')

rc_data = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50: continue
        try:
            name = line[0:11].strip()
            if not name: continue
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError): continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)
for name in rc_data:
    for key in rc_data[name]: rc_data[name][key] = np.array(rc_data[name][key])

sparc_props = {}
with open(mrt_path, 'r') as f: mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50: data_start = i + 1; break
for line in mrt_lines[data_start:]:
    if not line.strip() or line.startswith('#'): continue
    try:
        name = line[0:11].strip()
        parts = line[11:].split()
        if len(parts) < 17: continue
        sparc_props[name] = {'Inc': float(parts[4]), 'Vflat': float(parts[14]), 'Q': int(parts[16])}
    except: continue

sparc_r1 = []
sparc_ls_results = []
for name, gdata in rc_data.items():
    if name not in sparc_props: continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85: continue
    R = gdata['R']; Vobs = gdata['Vobs']; Vgas = gdata['Vgas']; Vdisk = gdata['Vdisk']; Vbul = gdata['Vbul']
    Vbar_sq = 0.5*Vdisk**2 + Vgas*np.abs(Vgas) + 0.7*Vbul*np.abs(Vbul)
    gbar_SI = np.where(R>0, np.abs(Vbar_sq)*(1e3)**2/(R*kpc_m), 1e-15)
    gobs_SI = np.where(R>0, (Vobs*1e3)**2/(R*kpc_m), 1e-15)
    valid = (gbar_SI>1e-15)&(gobs_SI>1e-15)&(R>0)&(Vobs>5)
    if np.sum(valid) < 15: continue
    Rv = R[valid]; si = np.argsort(Rv); Rs = Rv[si]
    lg = np.log10(gbar_SI[valid])[si]; lo = np.log10(gobs_SI[valid])[si]
    res = lo - rar_function(lg); res_dm = res - np.mean(res)
    r1 = lag_autocorrelation(res_dm, lag=1, center=True)
    if not np.isnan(r1):
        sparc_r1.append(r1)
        det, wl, pw, pp = run_ls_test(Rs, res_dm, rng=perm_rng)
        sparc_ls_results.append({'detected': det, 'wl_kpc': wl, 'perm_p': pp})

sparc_r1 = np.array(sparc_r1)
n_sparc = len(sparc_r1)
print(f"  SPARC: {n_sparc} galaxies (≥15 pts)")


# ══════════════════════════════════════════════════════════════════════
#  5. COMPARISON STATISTICS
# ══════════════════════════════════════════════════════════════════════
print(f"\n[5] Computing comparison statistics...")

tng_r1 = np.array([g['r1_dm'] for g in tng_galaxies])

print(f"\n  Lag-1 ACF (demeaned):")
print(f"    TNG (15 radii):  mean={np.mean(tng_r1):.4f} ± "
      f"{np.std(tng_r1,ddof=1)/np.sqrt(len(tng_r1)):.4f}, "
      f"median={np.median(tng_r1):.4f}, N={n_tng}")
print(f"    SPARC (full):    mean={np.mean(sparc_r1):.4f} ± "
      f"{np.std(sparc_r1,ddof=1)/np.sqrt(n_sparc):.4f}, "
      f"median={np.median(sparc_r1):.4f}, N={n_sparc}")

t_val, p_welch = stats.ttest_ind(tng_r1, sparc_r1, equal_var=False)
ks_stat, p_ks = stats.ks_2samp(tng_r1, sparc_r1)
print(f"    Welch p: {p_welch:.4e}, KS p: {p_ks:.4e}")

# Periodicity
tng_n_sig = sum(1 for g in tng_galaxies if g.get('ls_detected', False))
sparc_n_sig = sum(1 for r in sparc_ls_results if r['detected'])
tng_frac = tng_n_sig / n_tng if n_tng > 0 else 0
sparc_frac = sparc_n_sig / n_sparc if n_sparc > 0 else 0
_, fisher_p = stats.fisher_exact([[tng_n_sig, n_tng-tng_n_sig],
                                   [sparc_n_sig, n_sparc-sparc_n_sig]])

print(f"\n  Periodicity (p<0.05):")
print(f"    TNG:  {tng_n_sig}/{n_tng} ({tng_frac:.1%})")
print(f"    SPARC: {sparc_n_sig}/{n_sparc} ({sparc_frac:.1%})")
print(f"    Fisher p: {fisher_p:.4e}")

# Wavelengths
tng_wl = np.array([g['wl_kpc'] for g in tng_galaxies if g.get('ls_detected', False)])
sparc_wl = np.array([r['wl_kpc'] for r in sparc_ls_results if r['detected']])
if len(tng_wl) >= 3 and len(sparc_wl) >= 3:
    print(f"\n  Peak wavelength (sig. only):")
    print(f"    TNG:  median={np.median(tng_wl):.2f} kpc, "
          f"IQR={np.percentile(tng_wl,25):.2f}–{np.percentile(tng_wl,75):.2f}")
    print(f"    SPARC: median={np.median(sparc_wl):.2f} kpc, "
          f"IQR={np.percentile(sparc_wl,25):.2f}–{np.percentile(sparc_wl,75):.2f}")


# ══════════════════════════════════════════════════════════════════════
#  6. FIGURES
# ══════════════════════════════════════════════════════════════════════
print(f"\n[6] Generating figures...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

bins = np.linspace(-0.6, 1.0, 35)
ax = axes[0]
ax.hist(tng_r1, bins=bins, density=True, alpha=0.6, color='#4CAF50',
        label=f'TNG 15-radii (N={n_tng})', edgecolor='white', linewidth=0.5)
ax.hist(sparc_r1, bins=bins, density=True, alpha=0.6, color='#E91E63',
        label=f'SPARC (N={n_sparc})', edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(tng_r1), color='#2E7D32', ls='--', lw=1.5)
ax.axvline(np.mean(sparc_r1), color='#C2185B', ls='--', lw=1.5)
ax.set_xlabel('Lag-1 ACF (demeaned)')
ax.set_ylabel('Density')
ax.set_title('(a) ACF: TNG vs SPARC')
ax.legend(fontsize=7.5)

ax = axes[1]
cats = ['TNG\n(15 radii)', 'SPARC']
fracs = [tng_frac*100, sparc_frac*100]
bars = ax.bar(cats, fracs, color=['#4CAF50', '#E91E63'], alpha=0.8, width=0.5)
ax.axhline(5, color='grey', ls='--', lw=1)
for b, f, ns, nt in zip(bars, fracs, [tng_n_sig, sparc_n_sig], [n_tng, n_sparc]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{f:.1f}%\n({ns}/{nt})', ha='center', fontsize=9)
ax.set_ylabel('Fraction with periodicity (%)')
ax.set_title('(b) Periodicity Fraction')
ax.set_ylim(0, max(fracs)*1.3+5)

ax = axes[2]
for arr, lbl, clr in [(tng_r1, f'TNG (N={n_tng})', '#4CAF50'),
                        (sparc_r1, f'SPARC (N={n_sparc})', '#E91E63')]:
    s = np.sort(arr); c = np.arange(1, len(s)+1)/len(s)
    ax.plot(s, c, label=lbl, lw=1.8, color=clr)
ax.axvline(0, color='grey', ls=':', lw=0.8)
ax.set_xlabel('Lag-1 ACF (demeaned)')
ax.set_ylabel('CDF')
ax.set_title('(c) CDF Comparison')
ax.legend(fontsize=8)

fig.suptitle('TNG 15-Radii Coherence vs SPARC', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'tng_coherence_15radii.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════
#  7. SAVE JSON
# ══════════════════════════════════════════════════════════════════════
results = {
    'test': 'tng_coherence_15radii',
    'description': ('Coherence analysis on TNG100-1 with 15 log-spaced radii '
                    'per galaxy (0.5-5 × R_half). Compared to SPARC observations.'),
    'sample': {'tng_n': n_tng, 'sparc_n': n_sparc},
    'acf': {
        'tng_mean': round(float(np.mean(tng_r1)), 4),
        'tng_median': round(float(np.median(tng_r1)), 4),
        'tng_se': round(float(np.std(tng_r1, ddof=1)/np.sqrt(n_tng)), 4),
        'sparc_mean': round(float(np.mean(sparc_r1)), 4),
        'sparc_median': round(float(np.median(sparc_r1)), 4),
        'welch_p': float(p_welch),
        'ks_p': float(p_ks),
    },
    'periodicity': {
        'tng_n_sig': tng_n_sig, 'tng_frac': round(tng_frac, 4),
        'sparc_n_sig': sparc_n_sig, 'sparc_frac': round(sparc_frac, 4),
        'fisher_p': float(fisher_p),
    },
}

out_path = os.path.join(RESULTS_DIR, 'summary_tng_coherence_15radii.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: {out_path}")
print("=" * 76)
print("Done.")
