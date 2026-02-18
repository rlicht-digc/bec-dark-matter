#!/usr/bin/env python3
"""
PHANGS Inner Quantum Regime Test (X < 1)
==========================================
Uses PHANGS-ALMA CO rotation curves (Lang+2020) to probe the deep quantum
regime X = R/ξ < 1, where ξ = sqrt(GM*/g†) is the BEC healing length.

For massive spirals (M* > 10^10), ξ ~ 3-10 kpc. PHANGS data starts at
R = 0.125 kpc, giving X_min ~ 0.01-0.04 — well into the soliton core.

BEC prediction for X < 1:
  - Soliton core ρ(r) = ρ_c [1 + 0.091(r/r_c)²]^-8 dominates
  - RAR residuals should show systematic POSITIVE deviation (excess gobs)
  - The excess should scale with X in a specific functional form

CDM/NFW prediction for X < 1:
  - NFW profile is cuspy (ρ ~ r^-1) in the center
  - RAR residuals should be smaller (good NFW fit) or random

This test examines:
  1. Mean RAR residual at X < 1 vs X > 1
  2. Radial profile of residuals in X-coordinates
  3. Systematic excess test (is inner residual consistently positive?)
  4. Comparison of rising vs flat RC regimes

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import numpy as np
import os
import re
import json
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10  # m/s^2
G_kpc = 4.302e-6     # (km/s)^2 kpc / Msun
conv = 1e6 / 3.0857e19
G_SI = 6.674e-11     # m^3 kg^-1 s^-2
Msun = 1.989e30      # kg
kpc_m = 3.0857e19    # m per kpc

print("=" * 72)
print("PHANGS INNER QUANTUM REGIME TEST (X < 1)")
print("=" * 72)


def rar_pred(gbar):
    """McGaugh+2016 RAR."""
    gbar = np.asarray(gbar, dtype=float)
    x = np.sqrt(np.maximum(gbar, 1e-20) / g_dagger)
    d = 1.0 - np.exp(-x)
    d = np.maximum(d, 1e-20)
    return gbar / d


def compute_xi(logMs):
    """BEC healing length ξ = sqrt(GM*/g†) in kpc."""
    Ms = 10**logMs * Msun  # kg
    xi_m = np.sqrt(G_SI * Ms / g_dagger)  # meters
    return xi_m / kpc_m  # kpc


# ============================================================
# STEP 1: Load PHANGS rotation curves and properties
# ============================================================
print("\n[1] Loading PHANGS data...")

rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'phangs_lang2020_rotation_curves.tsv')
prop_path = os.path.join(DATA_DIR, 'hi_surveys', 'phangs_lang2020_properties.tsv')

# Properties
props = {}
with open(prop_path) as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        row = {header[i]: parts[i].strip().strip('"').strip() for i in range(min(len(header), len(parts)))}
        gid = row.get('ID', '').strip()
        if gid:
            try:
                props[gid] = {
                    'dist': float(row['Dist']),
                    'inc': float(row['i']),
                    'vsys': float(row['Vsys']),
                    'ra': float(row['RAJ2000']),
                    'dec': float(row['DEJ2000']),
                }
            except (ValueError, KeyError):
                pass

# Rotation curves grouped by galaxy
gal_rc = {}
with open(rc_path) as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        row = {header[i]: parts[i].strip().strip('"').strip() for i in range(min(len(header), len(parts)))}
        gid = row.get('ID', '').strip()
        if gid not in gal_rc:
            gal_rc[gid] = []
        try:
            gal_rc[gid].append({
                'R_kpc': float(row['Rad']),
                'V_rot': float(row['VRot']),
                'e_V_up': float(row.get('E_VRot', 'nan')),
                'e_V_dn': float(row.get('e_VRot', 'nan')),
            })
        except (ValueError, KeyError):
            pass

print(f"  Properties: {len(props)} galaxies")
print(f"  Rotation curves: {len(gal_rc)} galaxies")

# ============================================================
# STEP 2: Get WISE stellar masses (z0MGS)
# ============================================================
print("\n[2] Looking up WISE stellar masses...")

z0mgs_path = os.path.join(DATA_DIR, 'z0mgs_leroy2019_masses.tsv')
z0mgs = {}
if os.path.exists(z0mgs_path):
    with open(z0mgs_path) as f:
        h = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(h):
                continue
            row = {h[i]: parts[i].strip().strip('"') for i in range(len(h))}
            for prefix in ['NGC', 'IC', 'UGC']:
                raw = row.get(prefix, '').strip()
                if raw:
                    m = re.match(rf'^{prefix}0*(\d+\w*)', raw, re.IGNORECASE)
                    norm = f"{prefix}{m.group(1)}" if m else raw.upper().replace(' ', '')
                    try:
                        z0mgs[norm] = float(row.get('logM*', ''))
                    except (ValueError, TypeError):
                        pass

print(f"  z0MGS: {len(z0mgs)} galaxy masses")

# ============================================================
# STEP 3: Process each PHANGS galaxy
# ============================================================
print("\n[3] Processing PHANGS galaxies...")

galaxy_results = []

for gid in sorted(gal_rc.keys()):
    if gid not in props:
        continue

    p = props[gid]
    rc = gal_rc[gid]

    # Inclination cut
    if p['inc'] < 20 or p['inc'] > 80:
        continue

    # Normalize name
    norm = re.sub(r'\s+', '', gid)
    m = re.match(r'^(NGC|IC|UGC)0*(\d+\w*)$', norm, re.IGNORECASE)
    if m:
        norm = f"{m.group(1).upper()}{m.group(2)}"

    # Get stellar mass
    logMs = z0mgs.get(norm)
    if logMs is None:
        # Try Tully-Fisher fallback
        vflat = max([r['V_rot'] for r in rc])
        logMs = 3.75 * np.log10(max(vflat, 10)) + 2.0

    # Compute healing length
    xi_kpc = compute_xi(logMs)

    # Only interested in massive galaxies where ξ > 2 kpc
    if xi_kpc < 2.0:
        continue

    # Parse RC
    R_kpc = np.array([r['R_kpc'] for r in rc])
    V_rot = np.array([r['V_rot'] for r in rc])

    # Remove bad points
    good = (R_kpc > 0) & (V_rot > 0) & np.isfinite(R_kpc) & np.isfinite(V_rot)
    R_kpc = R_kpc[good]
    V_rot = V_rot[good]

    if len(R_kpc) < 5:
        continue

    # Compute X = R/ξ
    X = R_kpc / xi_kpc

    # gobs = V²/R in m/s²
    gobs = V_rot**2 / R_kpc * conv

    # Estimate gbar using exponential disk
    Mstar = 10**logMs
    M_gas = 0.33 * Mstar  # 33% gas fraction estimate
    M_total = Mstar + M_gas

    # Better disk model using CO extent
    mean_R = np.mean(R_kpc)
    Rd = mean_R / 2.2
    Rd = max(Rd, 0.3)

    log_res = []
    log_gbar = []
    X_pts = []
    R_pts = []

    for j in range(len(R_kpc)):
        r = R_kpc[j]
        x_rd = r / Rd
        M_enc = M_total * (1 - (1 + x_rd) * np.exp(-x_rd))
        gbar = G_kpc * M_enc / r**2 * conv

        if gbar <= 0 or gobs[j] <= 0:
            continue

        res = np.log10(gobs[j]) - np.log10(rar_pred(gbar))

        if abs(res) > 1.5:
            continue

        log_res.append(res)
        log_gbar.append(np.log10(gbar))
        X_pts.append(X[j])
        R_pts.append(r)

    if len(log_res) < 5:
        continue

    log_res = np.array(log_res)
    log_gbar = np.array(log_gbar)
    X_pts = np.array(X_pts)
    R_pts = np.array(R_pts)

    # Split at X = 1
    inner = X_pts < 1.0
    outer = X_pts >= 1.0
    n_inner = np.sum(inner)
    n_outer = np.sum(outer)

    galaxy_results.append({
        'name': norm,
        'logMs': logMs,
        'xi_kpc': xi_kpc,
        'dist': p['dist'],
        'n_pts': len(log_res),
        'n_inner': int(n_inner),
        'n_outer': int(n_outer),
        'mean_res_inner': float(np.mean(log_res[inner])) if n_inner > 0 else np.nan,
        'mean_res_outer': float(np.mean(log_res[outer])) if n_outer > 0 else np.nan,
        'mean_res_all': float(np.mean(log_res)),
        'std_res_all': float(np.std(log_res)),
        'log_res': log_res,
        'X_pts': X_pts,
        'R_pts': R_pts,
        'log_gbar': log_gbar,
        'Vmax': float(np.max(V_rot)),
    })

print(f"  Massive galaxies (ξ > 2 kpc): {len(galaxy_results)}")
print(f"  With inner points (X < 1): {sum(1 for g in galaxy_results if g['n_inner'] > 0)}")

# ============================================================
# STEP 4: Table of galaxies and their X coverage
# ============================================================
print("\n" + "=" * 72)
print("GALAXY TABLE: X = R/ξ Coverage")
print("=" * 72)

print(f"\n  {'Galaxy':>12s} {'logMs':>6s} {'ξ(kpc)':>7s} {'N_tot':>5s} {'N_X<1':>5s} "
      f"{'N_X>1':>5s} {'X_min':>6s} {'X_max':>6s} {'<res>_in':>8s} {'<res>_out':>8s}")
print(f"  {'-'*85}")

for g in sorted(galaxy_results, key=lambda x: -x['xi_kpc']):
    x_min = np.min(g['X_pts'])
    x_max = np.max(g['X_pts'])
    res_in = f"{g['mean_res_inner']:+.4f}" if not np.isnan(g['mean_res_inner']) else '   ---'
    res_out = f"{g['mean_res_outer']:+.4f}" if not np.isnan(g['mean_res_outer']) else '   ---'
    print(f"  {g['name']:>12s} {g['logMs']:6.2f} {g['xi_kpc']:7.2f} {g['n_pts']:5d} "
          f"{g['n_inner']:5d} {g['n_outer']:5d} {x_min:6.3f} {x_max:6.3f} {res_in:>8s} {res_out:>8s}")

# ============================================================
# STEP 5: Aggregate inner vs outer RAR residuals
# ============================================================
print("\n" + "=" * 72)
print("TEST 1: Inner (X < 1) vs Outer (X > 1) RAR Residuals")
print("=" * 72)

all_inner_res = np.concatenate([g['log_res'][g['X_pts'] < 1.0]
                                 for g in galaxy_results if np.any(g['X_pts'] < 1.0)])
all_outer_res = np.concatenate([g['log_res'][g['X_pts'] >= 1.0]
                                 for g in galaxy_results if np.any(g['X_pts'] >= 1.0)])

print(f"\n  Inner (X < 1): {len(all_inner_res)} points")
print(f"    Mean residual: {np.mean(all_inner_res):+.4f} dex")
print(f"    Median:        {np.median(all_inner_res):+.4f} dex")
print(f"    Scatter:       {np.std(all_inner_res):.4f} dex")

print(f"\n  Outer (X >= 1): {len(all_outer_res)} points")
print(f"    Mean residual: {np.mean(all_outer_res):+.4f} dex")
print(f"    Median:        {np.median(all_outer_res):+.4f} dex")
print(f"    Scatter:       {np.std(all_outer_res):.4f} dex")

delta = np.mean(all_inner_res) - np.mean(all_outer_res)
print(f"\n  Delta (inner - outer): {delta:+.4f} dex")

# Mann-Whitney test
from scipy.stats import mannwhitneyu
u_stat, p_mw = mannwhitneyu(all_inner_res, all_outer_res, alternative='two-sided')
print(f"  Mann-Whitney: U={u_stat:.0f}, p={p_mw:.6f}")

if delta > 0 and p_mw < 0.05:
    print("  -> Inner excess DETECTED (consistent with soliton core)")
elif p_mw > 0.05:
    print("  -> No significant difference (no soliton core signal)")
else:
    print("  -> Inner DEFICIT (opposite of BEC prediction)")

# ============================================================
# STEP 6: Per-galaxy inner excess test
# ============================================================
print("\n" + "=" * 72)
print("TEST 2: Per-Galaxy Inner Excess")
print("=" * 72)

n_inner_excess = 0
n_inner_deficit = 0
n_no_data = 0
per_gal_deltas = []

for g in galaxy_results:
    if g['n_inner'] < 3 or g['n_outer'] < 3:
        n_no_data += 1
        continue
    d = g['mean_res_inner'] - g['mean_res_outer']
    per_gal_deltas.append(d)
    if d > 0:
        n_inner_excess += 1
    else:
        n_inner_deficit += 1

per_gal_deltas = np.array(per_gal_deltas)
n_tested = n_inner_excess + n_inner_deficit

print(f"\n  Galaxies with inner excess:  {n_inner_excess}/{n_tested} ({100*n_inner_excess/max(n_tested,1):.0f}%)")
print(f"  Galaxies with inner deficit: {n_inner_deficit}/{n_tested} ({100*n_inner_deficit/max(n_tested,1):.0f}%)")
print(f"  (Insufficient data: {n_no_data})")

if n_tested >= 5:
    # Wilcoxon signed-rank test on per-galaxy deltas
    stat_w, p_w = wilcoxon(per_gal_deltas)
    print(f"\n  Mean per-galaxy delta: {np.mean(per_gal_deltas):+.4f} dex")
    print(f"  Wilcoxon signed-rank: W={stat_w:.1f}, p={p_w:.4f}")
    if p_w < 0.05 and np.mean(per_gal_deltas) > 0:
        print("  -> Systematic inner excess CONFIRMED")
    else:
        print("  -> No systematic inner excess")

# ============================================================
# STEP 7: Residual profile in X-coordinates
# ============================================================
print("\n" + "=" * 72)
print("TEST 3: Residual Profile in X = R/ξ Coordinates")
print("=" * 72)

all_X = np.concatenate([g['X_pts'] for g in galaxy_results])
all_res = np.concatenate([g['log_res'] for g in galaxy_results])

# Bin in log(X) space
logX_edges = np.arange(-2.0, 1.5, 0.3)
logX_centers = (logX_edges[:-1] + logX_edges[1:]) / 2

print(f"\n  {'log(X)':>8s} {'N':>6s} {'<res>':>8s} {'sigma':>8s}")
print(f"  {'-'*35}")

bin_means = []
bin_ns = []
for k in range(len(logX_centers)):
    mask = (np.log10(all_X) >= logX_edges[k]) & (np.log10(all_X) < logX_edges[k+1])
    n = np.sum(mask)
    if n >= 5:
        mu = np.mean(all_res[mask])
        sig = np.std(all_res[mask])
        print(f"  {logX_centers[k]:8.2f} {n:6d} {mu:+8.4f} {sig:8.4f}")
        bin_means.append(mu)
        bin_ns.append(n)
    else:
        print(f"  {logX_centers[k]:8.2f} {n:6d}      ---      ---")
        bin_means.append(np.nan)
        bin_ns.append(n)

# Spearman correlation of mean residual vs log(X)
valid_bins = ~np.isnan(bin_means)
if np.sum(valid_bins) >= 4:
    rho_X, p_X = spearmanr(np.array(logX_centers)[valid_bins],
                             np.array(bin_means)[valid_bins])
    print(f"\n  Spearman (mean_res vs log(X)): rho={rho_X:+.3f}, p={p_X:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Inner Quantum Regime Test")
print("=" * 72)

print(f"\n  Sample: {len(galaxy_results)} PHANGS galaxies with ξ > 2 kpc")
print(f"  X coverage: {np.min(all_X):.3f} to {np.max(all_X):.3f}")
print(f"  Points at X < 1: {len(all_inner_res)}")
print(f"  Points at X >= 1: {len(all_outer_res)}")
print(f"\n  Inner-outer delta: {delta:+.4f} dex (p={p_mw:.4f})")

if n_tested >= 5:
    print(f"  Per-galaxy Wilcoxon: p={p_w:.4f}")
    print(f"  Inner excess fraction: {n_inner_excess}/{n_tested}")

# BEC predicts inner EXCESS (delta > 0). Significant deficit is OPPOSITE.
bec_excess = delta > 0 and p_mw < 0.05
bec_deficit = delta < 0 and p_mw < 0.05
if bec_excess:
    verdict = "DETECTED — inner excess consistent with soliton core"
elif bec_deficit:
    verdict = "NOT DETECTED — significant inner DEFICIT (opposite of BEC prediction)"
else:
    verdict = "NOT DETECTED — no significant inner-outer difference"

# Also check per-galaxy direction
if n_tested >= 5:
    mean_delta = np.mean(per_gal_deltas)
    per_galaxy_verdict = "excess" if mean_delta > 0 else "deficit"
    print(f"  Per-galaxy mean delta: {mean_delta:+.4f} dex ({per_galaxy_verdict})")

bec_supported = bec_excess
print(f"\n  BEC soliton core signal: {verdict}")

# Save results
summary = {
    'test_name': 'phangs_inner_quantum_regime',
    'n_galaxies': len(galaxy_results),
    'n_inner_pts': int(len(all_inner_res)),
    'n_outer_pts': int(len(all_outer_res)),
    'X_range': [round(float(np.min(all_X)), 4), round(float(np.max(all_X)), 4)],
    'inner_mean_res': round(float(np.mean(all_inner_res)), 4),
    'outer_mean_res': round(float(np.mean(all_outer_res)), 4),
    'delta_inner_outer': round(float(delta), 4),
    'mann_whitney_p': round(float(p_mw), 6),
    'per_galaxy_excess_fraction': f"{n_inner_excess}/{n_tested}",
    'wilcoxon_p': round(float(p_w), 6) if n_tested >= 5 else None,
    'bec_signal_detected': bool(bec_supported),
    'galaxies': [{
        'name': g['name'],
        'logMs': round(g['logMs'], 2),
        'xi_kpc': round(g['xi_kpc'], 2),
        'n_inner': g['n_inner'],
        'n_outer': g['n_outer'],
        'mean_res_inner': round(g['mean_res_inner'], 4) if not np.isnan(g['mean_res_inner']) else None,
        'mean_res_outer': round(g['mean_res_outer'], 4) if not np.isnan(g['mean_res_outer']) else None,
    } for g in galaxy_results],
}

outpath = os.path.join(RESULTS_DIR, 'summary_phangs_inner_quantum.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
