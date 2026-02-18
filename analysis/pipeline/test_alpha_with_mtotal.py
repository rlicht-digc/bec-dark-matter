#!/usr/bin/env python3
"""
Alpha Rescaling Test with M_total vs M_star

The radial variance profile test found that the variance peaks at X = R/ξ_eff
where ξ_eff = α × √(GM*/g†) with α ≈ 3.5. This implies the true coherence
length is set by ~12× the stellar mass (since ξ ∝ √M, 3.5² ≈ 12).

The stellar-to-halo mass ratio M*/M_halo ≈ 0.01-0.1, so M_halo ≈ 10-100 × M*.
If the healing length is set by M_TOTAL (not just M_star), then using M_total
directly should give α ≈ 1.

This test:
  1. Gets halo masses from Kourkchi & Tully 2017 group catalog (logMh)
  2. For galaxies without group membership, estimates M_total via abundance matching
  3. Reruns the α scan with ξ = √(G·M_total/g†) and checks if peak lands at X ~ 1

If α drops from 3.5 → ~1 when switching from M* to M_total, that's a strong
confirmation that the condensation scale is set by total gravitating mass.
"""

import os
import sys
import json
import re
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
G_SI = 6.674e-11
Msun_kg = 1.989e30
kpc_m = 3.086e19
gdagger = 1.2e-10

def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)

def xi_from_mass(logM):
    """Healing length in kpc from any mass (stellar or total)."""
    M_SI = 10.0**logM * Msun_kg
    return np.sqrt(G_SI * M_SI / gdagger) / kpc_m


# ================================================================
# Abundance matching: M_star → M_halo (Behroozi+2013/Moster+2013)
# ================================================================
def moster2013_smhm(logMh):
    """Moster+2013 stellar-to-halo mass relation at z=0.
    Returns log10(M*/Msun) given log10(Mh/Msun)."""
    M1 = 11.59    # log10 characteristic halo mass
    N = 0.0351    # normalization
    beta = 1.376  # low-mass slope
    gamma = 0.608 # high-mass slope
    x = 10.0**(logMh - M1)
    ratio = 2.0 * N / (x**(-beta) + x**gamma)
    return logMh + np.log10(ratio)

def invert_smhm(logMs):
    """Invert the SMHM relation: given log10(M*), find log10(Mh).
    Uses bisection since the relation is monotonic."""
    lo, hi = 9.0, 16.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if moster2013_smhm(mid) < logMs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ================================================================
# Load SPARC data
# ================================================================
print("=" * 76)
print("ALPHA RESCALING TEST: M_total vs M_star")
print("Does α drop from ~3.5 to ~1 when using M_total?")
print("=" * 76)

print("\n[1] Loading SPARC rotation curves and properties...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

galaxies = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50: continue
        try:
            name = line[0:11].strip()
            if not name: continue
            dist = float(line[12:18].strip())
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            evobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except: continue
        if name not in galaxies:
            galaxies[name] = {'R':[], 'Vobs':[], 'eVobs':[], 'Vgas':[], 'Vdisk':[], 'Vbul':[], 'dist':dist}
        galaxies[name]['R'].append(rad)
        galaxies[name]['Vobs'].append(vobs)
        galaxies[name]['eVobs'].append(evobs)
        galaxies[name]['Vgas'].append(vgas)
        galaxies[name]['Vdisk'].append(vdisk)
        galaxies[name]['Vbul'].append(vbul)

for name in galaxies:
    for key in ['R','Vobs','eVobs','Vgas','Vdisk','Vbul']:
        galaxies[name][key] = np.array(galaxies[name][key])

with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break

sparc_props = {}
for line in mrt_lines[data_start:]:
    if not line.strip() or line.startswith('#'): continue
    try:
        name = line[0:11].strip()
        if not name: continue
        parts = line[11:].split()
        if len(parts) < 17: continue
        Inc = float(parts[4])
        L36 = float(parts[6])
        Q = int(parts[16])
        logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
        sparc_props[name] = {'Inc': Inc, 'Q': Q, 'logMs': logMs}
    except: continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with properties")


# ================================================================
# Load Kourkchi & Tully 2017 group catalog for halo masses
# ================================================================
print("\n[2] Loading Kourkchi & Tully 2017 group catalog...")

k17_gal_path = os.path.join(DATA_DIR, 'kourkchi2017_galaxies.tsv')
k17_grp_path = os.path.join(DATA_DIR, 'kourkchi2017_massive_groups.tsv')

k17_halo_masses = {}  # galaxy_name → logMh

if os.path.exists(k17_gal_path) and os.path.exists(k17_grp_path):
    # Load group properties
    group_props = {}
    with open(k17_grp_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header): continue
            row = {header[i]: parts[i].strip() for i in range(len(header))}
            pgc1 = row.get('PGC1', '').strip()
            try:
                logMd = float(row.get('logMd', '0') or '0')
            except: continue
            if pgc1 and logMd > 0:
                group_props[pgc1] = logMd

    # Load galaxy→group assignments
    with open(k17_gal_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header): continue
            row = {header[i]: parts[i].strip() for i in range(len(header))}
            name_raw = row.get('Name', '').strip()
            pgc1 = row.get('PGC1', '').strip()
            if name_raw and pgc1 and pgc1 in group_props:
                norm = re.sub(r'\s+', '', name_raw.upper())
                k17_halo_masses[norm] = group_props[pgc1]

    print(f"  K17 groups loaded: {len(group_props)} groups, {len(k17_halo_masses)} galaxy→halo assignments")
else:
    print(f"  K17 catalog not found, will use abundance matching only")


# Known structure halo masses (from the unified pipeline)
STRUCTURES = {
    'Virgo': (187.71, 12.39, 1100, 14.9, 6.0, 800),
    'Fornax': (54.62, -35.45, 1379, 14.0, 2.0, 370),
    'UMa': (178.0, 49.0, 1050, 12.8, 8.0, 200),
    'M81': (148.89, 69.07, -34, 12.0, 3.0, 200),
    'Sculptor': (11.89, -33.72, 200, 11.5, 5.0, 100),
    'CenA': (201.37, -43.02, 547, 12.5, 5.0, 200),
}

# UMa membership
UMA_GALAXIES = {
    'NGC3726', 'NGC3769', 'NGC3877', 'NGC3893', 'NGC3917',
    'NGC3949', 'NGC3953', 'NGC3972', 'NGC3992', 'NGC4010',
    'NGC4013', 'NGC4051', 'NGC4085', 'NGC4088', 'NGC4100',
    'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
    'UGC06399', 'UGC06446', 'UGC06667', 'UGC06786', 'UGC06787',
    'UGC06818', 'UGC06917', 'UGC06923', 'UGC06930', 'UGC06973',
    'UGC06983', 'UGC07089',
}

GROUP_MEMBERS = {
    'NGC2403': ('M81', 12.0), 'NGC2976': ('M81', 12.0), 'IC2574': ('M81', 12.0),
    'DDO154': ('M81', 12.0), 'DDO168': ('M81', 12.0),
    'NGC0300': ('Sculptor', 11.5), 'NGC0055': ('Sculptor', 11.5),
    'NGC7793': ('Sculptor', 11.5),
    'NGC5055': ('M101', 12.3),
}


# ================================================================
# Assign M_total to each SPARC galaxy
# ================================================================
print("\n[3] Assigning M_total to each SPARC galaxy...")

galaxy_mtotal = {}  # name → logMtotal
method_counts = {'K17': 0, 'structure': 0, 'abundance_matching': 0}

for name, prop in sparc_props.items():
    logMs = prop['logMs']

    # Method 1: K17 group catalog (direct halo mass measurement)
    norm = re.sub(r'\s+', '', name.upper())
    if norm in k17_halo_masses:
        logMh = k17_halo_masses[norm]
        galaxy_mtotal[name] = logMh
        method_counts['K17'] += 1
        continue

    # Method 2: Known structure membership
    if name in UMA_GALAXIES:
        galaxy_mtotal[name] = 12.8  # UMa halo mass
        method_counts['structure'] += 1
        continue
    if name in GROUP_MEMBERS:
        galaxy_mtotal[name] = GROUP_MEMBERS[name][1]
        method_counts['structure'] += 1
        continue

    # Method 3: Abundance matching (Moster+2013 SMHM)
    logMh_am = invert_smhm(logMs)
    galaxy_mtotal[name] = logMh_am
    method_counts['abundance_matching'] += 1

print(f"  K17 catalog: {method_counts['K17']} galaxies")
print(f"  Structure membership: {method_counts['structure']} galaxies")
print(f"  Abundance matching: {method_counts['abundance_matching']} galaxies")
print(f"  Total: {sum(method_counts.values())} galaxies")

# Show some examples
print(f"\n  {'Galaxy':15s} {'logM*':>7s} {'logMh':>7s} {'M*/Mh':>8s} {'ξ_star':>8s} {'ξ_halo':>8s} {'ratio':>7s}")
print(f"  {'-'*65}")
examples = ['NGC2403', 'NGC3198', 'NGC6503', 'NGC2841', 'UGC06786', 'DDO154']
for name in examples:
    if name in sparc_props and name in galaxy_mtotal:
        logMs = sparc_props[name]['logMs']
        logMh = galaxy_mtotal[name]
        ratio_m = 10**(logMs - logMh)
        xi_s = xi_from_mass(logMs)
        xi_h = xi_from_mass(logMh)
        xi_ratio = xi_h / xi_s
        print(f"  {name:15s} {logMs:>7.2f} {logMh:>7.2f} {ratio_m:>8.4f} {xi_s:>8.2f} {xi_h:>8.2f} {xi_ratio:>7.2f}")


# ================================================================
# Build RAR points
# ================================================================
print("\n[4] Computing RAR residuals...")

all_points = []
galaxy_names_used = set()

for name, gdata in galaxies.items():
    if name not in sparc_props or name not in galaxy_mtotal: continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85: continue

    R = gdata['R']; Vobs = gdata['Vobs']; eVobs = gdata['eVobs']
    Vdisk = gdata['Vdisk']; Vgas = gdata['Vgas']; Vbul = gdata['Vbul']

    Vbar_sq = 0.5*Vdisk**2 + Vgas*np.abs(Vgas) + 0.7*Vbul*np.abs(Vbul)
    gbar_SI = np.where(R > 0, (np.sqrt(np.abs(Vbar_sq))*1e3)**2 / (R*kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs*1e3)**2 / (R*kpc_m), 1e-15)
    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 0) & (eVobs/np.maximum(Vobs,1) < 0.3)
    if np.sum(valid) < 3: continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_res = log_gobs - rar_function(log_gbar)

    logMs = prop['logMs']
    logMh = galaxy_mtotal[name]
    xi_star = xi_from_mass(logMs)
    xi_halo = xi_from_mass(logMh)
    galaxy_names_used.add(name)

    for i in range(len(log_gbar)):
        all_points.append({
            'galaxy': name,
            'log_gbar': float(log_gbar[i]),
            'log_res': float(log_res[i]),
            'R_kpc': float(R[valid][i]),
            'logMs': logMs,
            'logMh': logMh,
            'xi_star': xi_star,
            'xi_halo': xi_halo,
        })

# Z-score
all_res = np.array([p['log_res'] for p in all_points])
mu, std = np.mean(all_res), np.std(all_res)
for p in all_points:
    p['z_res'] = (p['log_res'] - mu) / std

print(f"  {len(all_points)} points from {len(galaxy_names_used)} galaxies")


# ================================================================
# MAIN TEST: α scan with M_star vs M_total
# ================================================================
print("\n" + "=" * 76)
print("COMPARISON: α SCAN WITH M_star vs M_total")
print("=" * 76)

def peak_model(X, A, Xp, C):
    return A * X * np.exp(-X / max(Xp, 0.01)) + C

def run_alpha_scan(points, xi_key, label, alphas):
    """Run the α scan for a given healing length source."""
    results = []
    for alpha in alphas:
        for p in points:
            xi_eff = alpha * p[xi_key]
            p['X_scan'] = p['R_kpc'] / xi_eff if xi_eff > 0 else 999.0

        logX_edges = np.arange(-1.5, 2.5, 0.3)
        bX = []; bV = []; bVe = []
        for j in range(len(logX_edges) - 1):
            lo, hi = logX_edges[j], logX_edges[j+1]
            z_vals = np.array([p['z_res'] for p in points
                               if lo <= np.log10(max(p['X_scan'], 1e-5)) < hi])
            X_vals = np.array([p['X_scan'] for p in points
                               if lo <= np.log10(max(p['X_scan'], 1e-5)) < hi])
            if len(z_vals) >= 25:
                bX.append(float(np.median(X_vals)))
                bV.append(float(np.var(z_vals)))
                bVe.append(np.sqrt(2.0*float(np.var(z_vals))**2/(len(z_vals)-1)))

        bX = np.array(bX); bV = np.array(bV); bVe = np.array(bVe)
        if len(bV) < 4: continue

        try:
            pp, _ = curve_fit(peak_model, bX, bV, p0=[0.5, 1.0, 0.5],
                              sigma=bVe, absolute_sigma=True,
                              bounds=([0, 0.01, 0], [50, 50, 5]), maxfev=10000)

            # Constant model for comparison
            wt = 1.0 / np.maximum(bVe, 1e-6)
            K = np.average(bV, weights=wt**2)
            chi2_k = np.sum(((bV - K) / bVe)**2)
            aic_k = chi2_k + 2*1

            chi2_p = np.sum(((bV - peak_model(bX, *pp)) / bVe)**2)
            aic_p = chi2_p + 2*3

            results.append({
                'alpha': alpha,
                'X_peak': float(pp[1]),
                'A': float(pp[0]),
                'C': float(pp[2]),
                'aic_peak': float(aic_p),
                'aic_const': float(aic_k),
                'daic': float(aic_k - aic_p),
                'n_bins': len(bV),
            })
        except:
            continue
    return results

alphas = np.arange(0.25, 12.1, 0.25)

print(f"\n  Scanning α from {alphas[0]} to {alphas[-1]} in steps of 0.25...")

# Scan with M_star
print(f"\n  --- Using M_STAR (ξ = α × √(GM*/g†)) ---")
res_mstar = run_alpha_scan(all_points, 'xi_star', 'M_star', alphas)

# Scan with M_total
print(f"  --- Using M_TOTAL (ξ = α × √(GM_total/g†)) ---")
res_mtotal = run_alpha_scan(all_points, 'xi_halo', 'M_total', alphas)

# Find α where X_peak = 1 for each
def find_crossing(results, target=1.0):
    """Find α where X_peak crosses target via linear interpolation."""
    for i in range(len(results) - 1):
        x1, x2 = results[i]['X_peak'], results[i+1]['X_peak']
        a1, a2 = results[i]['alpha'], results[i+1]['alpha']
        if (x1 - target) * (x2 - target) <= 0:
            frac = (target - x1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else 0.5
            return a1 + frac * (a2 - a1)
    return None

alpha_cross_mstar = find_crossing(res_mstar)
alpha_cross_mtotal = find_crossing(res_mtotal)

print(f"\n{'=' * 76}")
print(f"RESULTS")
print(f"{'=' * 76}")

# Print table header
print(f"\n  {'alpha':>6} {'X_peak(M*)':>12} {'X_peak(Mh)':>12} {'ΔAIC(M*)':>10} {'ΔAIC(Mh)':>10}")
print(f"  {'-'*55}")

# Merge results at matching alphas
mstar_dict = {r['alpha']: r for r in res_mstar}
mtotal_dict = {r['alpha']: r for r in res_mtotal}
common_alphas = sorted(set(mstar_dict.keys()) & set(mtotal_dict.keys()))

for a in common_alphas:
    if a % 0.5 == 0 or a in [3.5, 3.25, 3.75]:  # print at regular intervals
        ms = mstar_dict[a]
        mt = mtotal_dict[a]
        flag_ms = ' <--' if ms['X_peak'] and 0.7 < ms['X_peak'] < 1.3 else ''
        flag_mt = ' <--' if mt['X_peak'] and 0.7 < mt['X_peak'] < 1.3 else ''
        print(f"  {a:>6.2f} {ms['X_peak']:>12.2f}{flag_ms:4s} {mt['X_peak']:>12.2f}{flag_mt:4s} "
              f"{ms['daic']:>+10.2f} {mt['daic']:>+10.2f}")

print(f"\n  X_peak = 1 crossing:")
if alpha_cross_mstar:
    print(f"    M_star:  α = {alpha_cross_mstar:.2f}")
else:
    print(f"    M_star:  no crossing found")
if alpha_cross_mtotal:
    print(f"    M_total: α = {alpha_cross_mtotal:.2f}")
else:
    print(f"    M_total: no crossing found")

# THE KEY RESULT
print(f"\n  {'=' * 60}")
if alpha_cross_mstar and alpha_cross_mtotal:
    ratio = alpha_cross_mstar / alpha_cross_mtotal
    print(f"  α(M*) / α(Mh) = {alpha_cross_mstar:.2f} / {alpha_cross_mtotal:.2f} = {ratio:.2f}")
    if 0.5 < alpha_cross_mtotal < 2.0:
        print(f"\n  >>> α drops to ~{alpha_cross_mtotal:.1f} with M_total!")
        print(f"  >>> The healing length IS set by total gravitating mass.")
        print(f"  >>> ξ = √(G·M_total/g†) with no extra scale factor needed.")
    elif alpha_cross_mtotal < alpha_cross_mstar:
        print(f"\n  >>> α decreased from {alpha_cross_mstar:.2f} to {alpha_cross_mtotal:.2f}")
        print(f"  >>> Moves in right direction but doesn't reach 1.")
        print(f"  >>> M_total may be underestimated (abundance matching floor).")
    else:
        print(f"\n  >>> α did not decrease with M_total.")
        print(f"  >>> Coherence scale may not be set by total mass alone.")

# Compute median M*/Mh ratio for the sample
ratios = []
for name in galaxy_names_used:
    if name in sparc_props and name in galaxy_mtotal:
        logMs = sparc_props[name]['logMs']
        logMh = galaxy_mtotal[name]
        ratios.append(10**(logMs - logMh))
if ratios:
    med_ratio = np.median(ratios)
    print(f"\n  Sample median M*/Mh = {med_ratio:.4f} (= 1/{1/med_ratio:.0f})")
    print(f"  Expected α(M*)/α(Mh) = √(Mh/M*) = {np.sqrt(1/med_ratio):.2f}")
    if alpha_cross_mstar and alpha_cross_mtotal:
        print(f"  Observed α(M*)/α(Mh) = {ratio:.2f}")


# ================================================================
# PLOT
# ================================================================
print("\n[5] Generating plots...")
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: X_peak(α) for both M_star and M_total
    ax1 = axes[0]
    a_ms = [r['alpha'] for r in res_mstar]
    xp_ms = [r['X_peak'] for r in res_mstar]
    a_mt = [r['alpha'] for r in res_mtotal]
    xp_mt = [r['X_peak'] for r in res_mtotal]
    ax1.plot(a_ms, xp_ms, 'b.-', markersize=6, lw=1.5, label='Using M_star')
    ax1.plot(a_mt, xp_mt, 'r.-', markersize=6, lw=1.5, label='Using M_total')
    ax1.axhline(1.0, color='black', ls='--', lw=2, alpha=0.5, label='X_peak = 1 (theory)')
    if alpha_cross_mstar:
        ax1.axvline(alpha_cross_mstar, color='blue', ls=':', lw=1.5, alpha=0.7,
                    label=f'α(M*) = {alpha_cross_mstar:.2f}')
    if alpha_cross_mtotal:
        ax1.axvline(alpha_cross_mtotal, color='red', ls=':', lw=1.5, alpha=0.7,
                    label=f'α(Mh) = {alpha_cross_mtotal:.2f}')
    ax1.set_xlabel('Scale factor α', fontsize=13)
    ax1.set_ylabel('Fitted X_peak', fontsize=13)
    ax1.set_title('Healing Length: M_star vs M_total', fontsize=14)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(-0.5, 8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: σ²(X) at α=1 with M_total
    ax2 = axes[1]
    for p in all_points:
        p['X_mt1'] = p['R_kpc'] / p['xi_halo'] if p['xi_halo'] > 0 else 999.0
    logX_edges = np.arange(-1.5, 2.5, 0.3)
    bx = []; bv = []; bve = []
    for j in range(len(logX_edges) - 1):
        lo, hi = logX_edges[j], logX_edges[j+1]
        z_vals = np.array([p['z_res'] for p in all_points if lo <= np.log10(max(p['X_mt1'],1e-5)) < hi])
        x_vals = np.array([p['X_mt1'] for p in all_points if lo <= np.log10(max(p['X_mt1'],1e-5)) < hi])
        if len(z_vals) >= 25:
            bx.append(float(np.median(x_vals))); bv.append(float(np.var(z_vals)))
            bve.append(np.sqrt(2.0*float(np.var(z_vals))**2/(len(z_vals)-1)))
    ax2.errorbar(bx, bv, yerr=bve, fmt='ro', markersize=7, capsize=4, zorder=5)
    ax2.axvline(1.0, color='red', ls='--', lw=2, alpha=0.7, label='X = 1')
    ax2.set_xscale('log')
    ax2.set_xlabel('X = R / ξ(M_total)', fontsize=13)
    ax2.set_ylabel('Variance σ²(Z-res)', fontsize=13)
    ax2.set_title('σ²(X) with M_total at α = 1', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: M*/Mh distribution
    ax3 = axes[2]
    log_ratios = [sparc_props[n]['logMs'] - galaxy_mtotal[n]
                  for n in galaxy_names_used
                  if n in sparc_props and n in galaxy_mtotal]
    ax3.hist(log_ratios, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(np.median(log_ratios), color='red', ls='--', lw=2,
                label=f'Median log(M*/Mh) = {np.median(log_ratios):.2f}')
    ax3.set_xlabel('log₁₀(M_star / M_halo)', fontsize=13)
    ax3.set_ylabel('Count', fontsize=13)
    ax3.set_title('Stellar-to-Halo Mass Ratio', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'alpha_mstar_vs_mtotal.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
except Exception as e:
    print(f"  Plot error: {e}")
    import traceback; traceback.print_exc()

# Save results
output = {
    'alpha_cross_mstar': float(alpha_cross_mstar) if alpha_cross_mstar else None,
    'alpha_cross_mtotal': float(alpha_cross_mtotal) if alpha_cross_mtotal else None,
    'n_galaxies': len(galaxy_names_used),
    'n_points': len(all_points),
    'method_counts': method_counts,
    'median_log_ms_over_mh': float(np.median(log_ratios)) if log_ratios else None,
    'scan_mstar': res_mstar,
    'scan_mtotal': res_mtotal,
}
with open(os.path.join(OUTPUT_DIR, 'summary_alpha_mtotal.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 76}")
print("Done.")
