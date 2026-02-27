#!/usr/bin/env python3
"""
Brouwer+2021 Weak Lensing RAR Scatter Analysis
=================================================

Different physical effect (gravitational lensing), same prediction.

The Brouwer+2021 KiDS-1000 data provides the lensing RAR: gbar vs g_obs
measured through weak gravitational lensing of ~1,000 sq degrees.

We have:
  - Isolated galaxy RAR (no bins) — Fig 4-5
  - ALL galaxy RAR by mass bin (4 bins) — Fig A4
  - Isolated galaxy RAR by mass bin (4 bins) — Fig 9
  - Isolated galaxy RAR by color (2 bins) — Fig 8
  - Isolated galaxy RAR by Sersic index (2 bins) — Fig 8
  - Isolated dwarf galaxy RAR — Fig 10
  - GAMA isolated galaxy RAR — Fig 4

Tests:
  1. Mass-binned scatter: do different mass bins (acceleration regimes)
     show different scatter relative to the canonical RAR?
  2. Isolated vs ALL: environment proxy. Do non-isolated galaxies show
     more scatter at certain accelerations?
  3. Color/Sersic split: different galaxy types as environment proxy.
  4. Does the lensing RAR residual scatter show an inversion near g†?

The key insight: lensing measures g_obs directly from light bending.
No rotation curves, no distance-dependent radii, no M/L assumptions.
If the same phase transition appears here, it's not a SPARC artifact.
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'brouwer2021')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
LOG_G_DAGGER = np.log10(g_dagger)
G_pc3 = 4.52e-30  # G in pc^3/(Msun*s^2)
pc_per_m = 3.086e16


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR prediction."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def load_esd_file(filename):
    """
    Load a Brouwer ESD profile.
    Returns gbar (m/s²), ESD_t (corrected), error (corrected).
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  WARNING: {filename} not found")
        return None

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gbar_acc = float(parts[0])    # m/s² (this IS gbar)
                esd_t = float(parts[1])       # h70*Msun/pc²
                esd_x = float(parts[2])       # cross (sanity check)
                error = float(parts[3])       # h70*Msun/pc²
                bias = float(parts[4])        # multiplicative calibration

                # Apply bias correction
                esd_corrected = esd_t / bias
                err_corrected = error / bias

                data.append({
                    'gbar': gbar_acc,
                    'log_gbar': np.log10(gbar_acc) if gbar_acc > 0 else -99,
                    'esd_t': esd_corrected,
                    'error': err_corrected,
                })
            except (ValueError, IndexError):
                continue

    return data


def esd_to_gobs(esd_t):
    """Convert ESD (h70*Msun/pc²) to g_obs (m/s²).
    g_obs = 4 * G * ESD_t * [pc/m]
    """
    return 4.0 * G_pc3 * esd_t * pc_per_m


def compute_lensing_rar_residuals(data):
    """Compute log(g_obs) - RAR_prediction(log(gbar)) for each point."""
    results = []
    for d in data:
        if d['log_gbar'] < -16 or d['esd_t'] <= 0:
            continue

        gobs = esd_to_gobs(d['esd_t'])
        if gobs <= 0:
            continue

        log_gobs = np.log10(gobs)
        log_gobs_pred = rar_function(d['log_gbar'])
        residual = log_gobs - log_gobs_pred

        # Error propagation
        gobs_err = esd_to_gobs(d['error'])
        log_gobs_err = gobs_err / (gobs * np.log(10)) if gobs > 0 else 1.0

        results.append({
            'log_gbar': d['log_gbar'],
            'log_gobs': log_gobs,
            'log_gobs_pred': log_gobs_pred,
            'residual': residual,
            'error': log_gobs_err,
        })

    return results


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("BROUWER+2021 WEAK LENSING RAR ANALYSIS")
print("=" * 72)

# ================================================================
# Load all datasets
# ================================================================
datasets = {}

# Main isolated RAR
datasets['isolated'] = load_esd_file('Fig-4-5-C1_RAR-KiDS-isolated_Nobins.txt')

# GAMA isolated
datasets['gama_isolated'] = load_esd_file('Fig-4-C1_RAR-GAMA-isolated_Nobins.txt')

# Isolated dwarfs
datasets['isolated_dwarfs'] = load_esd_file('Fig-10_RAR-KiDS-isolated-dwarfs_Nobins.txt')

# Mass-binned isolated (Fig 9)
for i in range(1, 5):
    datasets[f'isolated_mass{i}'] = load_esd_file(f'Fig-9_RAR-KiDS-isolated_Massbin-{i}.txt')

# Mass-binned ALL galaxies (Fig A4)
for i in range(1, 5):
    datasets[f'all_mass{i}'] = load_esd_file(f'Fig-A4_RAR-KiDS-all_Massbin-{i}.txt')

# Color-binned isolated (Fig 8)
for i in range(1, 3):
    datasets[f'isolated_color{i}'] = load_esd_file(f'Fig-8_RAR-KiDS-isolated_Colorbin_{i}.txt')

# Sersic-binned isolated (Fig 8)
for i in range(1, 3):
    datasets[f'isolated_sersic{i}'] = load_esd_file(f'Fig-8_RAR-KiDS-isolated_Sersicbin_{i}.txt')

# Hot gas corrected
datasets['isolated_hotgas'] = load_esd_file('Fig-4_RAR-KiDS-isolated_hotgas_Nobins.txt')

print(f"\n  Loaded {len(datasets)} datasets")
for name, data in datasets.items():
    if data:
        print(f"    {name}: {len(data)} points")

# Mass bin limits: log10(M*/(h70^-2 Msun)) = [8.5, 10.3, 10.6, 10.8, 11.0]
mass_bin_labels = {
    1: '8.5-10.3 (dwarfs)',
    2: '10.3-10.6',
    3: '10.6-10.8',
    4: '10.8-11.0 (massive)',
}

# ================================================================
# TEST 1: Lensing RAR residuals for each dataset
# ================================================================
print(f"\n{'='*72}")
print("TEST 1: LENSING RAR RESIDUALS")
print(f"{'='*72}")

all_residuals = {}
print(f"\n  {'Dataset':30s} {'N_pts':>6s} {'mean_res':>10s} {'σ_res':>8s} {'χ²/dof':>8s}")
print(f"  {'-'*66}")

for name, data in datasets.items():
    if not data:
        continue

    res = compute_lensing_rar_residuals(data)
    if len(res) < 3:
        continue

    all_residuals[name] = res

    residuals = np.array([r['residual'] for r in res])
    errors = np.array([r['error'] for r in res])

    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    chi2 = np.sum((residuals / errors)**2) if np.all(errors > 0) else np.nan
    dof = max(len(residuals) - 1, 1)

    print(f"  {name:30s} {len(res):6d} {mean_r:+10.4f} {std_r:8.4f} {chi2/dof:8.2f}")

# ================================================================
# TEST 2: ISOLATED vs ALL scatter comparison (environment)
# ================================================================
print(f"\n{'='*72}")
print("TEST 2: ISOLATED vs ALL GALAXIES (ENVIRONMENT PROXY)")
print(f"{'='*72}")
print("  Mass bin limits: log10(M*/(h70^-2 Msun)) = [8.5, 10.3, 10.6, 10.8, 11.0]")

print(f"\n  {'Mass bin':>25s} {'σ_isolated':>12s} {'σ_all':>10s} {'Δσ(all-iso)':>12s}")
print(f"  {'-'*62}")

iso_vs_all = []
for i in range(1, 5):
    iso_key = f'isolated_mass{i}'
    all_key = f'all_mass{i}'

    if iso_key not in all_residuals or all_key not in all_residuals:
        continue

    iso_res = np.array([r['residual'] for r in all_residuals[iso_key]])
    all_res_arr = np.array([r['residual'] for r in all_residuals[all_key]])

    s_iso = np.std(iso_res)
    s_all = np.std(all_res_arr)
    delta = s_all - s_iso

    iso_vs_all.append({
        'mass_bin': i,
        'label': mass_bin_labels[i],
        'sigma_isolated': round(s_iso, 5),
        'sigma_all': round(s_all, 5),
        'delta': round(delta, 5),
        'n_iso': len(iso_res),
        'n_all': len(all_res_arr),
    })

    print(f"  {mass_bin_labels[i]:>25s} {s_iso:12.4f} {s_all:10.4f} {delta:+12.4f}")

# Is the pattern: small delta at low mass (condensate), large delta at high mass (baryon)?
if len(iso_vs_all) >= 2:
    low_delta = iso_vs_all[0]['delta'] if iso_vs_all else 0
    high_delta = iso_vs_all[-1]['delta'] if iso_vs_all else 0
    print(f"\n  Low-mass Δσ = {low_delta:+.4f}, High-mass Δσ = {high_delta:+.4f}")
    if abs(low_delta) < abs(high_delta):
        print("  → Environment matters MORE at high mass (baryon regime) — BEC-consistent")
    else:
        print("  → Environment matters MORE at low mass — opposite to BEC prediction")

# ================================================================
# TEST 3: COLOR SPLIT (environment/morphology proxy)
# ================================================================
print(f"\n{'='*72}")
print("TEST 3: BLUE vs RED GALAXIES (COLOR SPLIT)")
print(f"{'='*72}")

if 'isolated_color1' in all_residuals and 'isolated_color2' in all_residuals:
    blue_res = np.array([r['residual'] for r in all_residuals['isolated_color1']])
    red_res = np.array([r['residual'] for r in all_residuals['isolated_color2']])

    print(f"  Blue (u-r low):  σ = {np.std(blue_res):.4f} dex  (N={len(blue_res)})")
    print(f"  Red (u-r high):  σ = {np.std(red_res):.4f} dex  (N={len(red_res)})")
    print(f"  Δσ(red-blue) = {np.std(red_res) - np.std(blue_res):+.4f}")

# ================================================================
# TEST 4: SERSIC SPLIT (morphology proxy)
# ================================================================
print(f"\n{'='*72}")
print("TEST 4: LOW vs HIGH SERSIC INDEX")
print(f"{'='*72}")

if 'isolated_sersic1' in all_residuals and 'isolated_sersic2' in all_residuals:
    low_n_res = np.array([r['residual'] for r in all_residuals['isolated_sersic1']])
    high_n_res = np.array([r['residual'] for r in all_residuals['isolated_sersic2']])

    print(f"  Low Sersic (disk):     σ = {np.std(low_n_res):.4f} dex  (N={len(low_n_res)})")
    print(f"  High Sersic (bulge):   σ = {np.std(high_n_res):.4f} dex  (N={len(high_n_res)})")
    print(f"  Δσ(high-low) = {np.std(high_n_res) - np.std(low_n_res):+.4f}")

# ================================================================
# TEST 5: ACCELERATION-BINNED SCATTER (look for g† inversion)
# ================================================================
print(f"\n{'='*72}")
print("TEST 5: ACCELERATION-BINNED LENSING RAR SCATTER")
print(f"{'='*72}")

# Combine all isolated mass-bin data for per-acceleration analysis
combined_iso = []
for i in range(1, 5):
    key = f'isolated_mass{i}'
    if key in all_residuals:
        for r in all_residuals[key]:
            combined_iso.append({**r, 'mass_bin': i})

combined_all = []
for i in range(1, 5):
    key = f'all_mass{i}'
    if key in all_residuals:
        for r in all_residuals[key]:
            combined_all.append({**r, 'mass_bin': i})

if combined_iso:
    iso_gbar = np.array([r['log_gbar'] for r in combined_iso])
    iso_resid = np.array([r['residual'] for r in combined_iso])

    all_gbar = np.array([r['log_gbar'] for r in combined_all]) if combined_all else np.array([])
    all_resid = np.array([r['residual'] for r in combined_all]) if combined_all else np.array([])

    # Bin by acceleration
    acc_edges = np.linspace(-15, -11, 9)
    acc_centers = (acc_edges[:-1] + acc_edges[1:]) / 2

    print(f"\n  {'log(gbar)':>10s} {'N_iso':>6s} {'σ_iso':>8s} {'N_all':>6s} {'σ_all':>8s} {'Δσ':>8s}")
    print(f"  {'-'*52}")

    acc_profile = []
    for j in range(len(acc_edges)-1):
        lo, hi = acc_edges[j], acc_edges[j+1]
        center = acc_centers[j]

        iso_mask = (iso_gbar >= lo) & (iso_gbar < hi)
        all_mask = (all_gbar >= lo) & (all_gbar < hi) if len(all_gbar) > 0 else np.array([])

        iso_pts = iso_resid[iso_mask]
        all_pts = all_resid[all_mask] if len(all_resid) > 0 else np.array([])

        s_iso = np.std(iso_pts) if len(iso_pts) >= 3 else np.nan
        s_all = np.std(all_pts) if len(all_pts) >= 3 else np.nan
        delta = s_all - s_iso if not (np.isnan(s_iso) or np.isnan(s_all)) else np.nan

        acc_profile.append({
            'center': round(center, 2),
            'n_iso': len(iso_pts), 'n_all': len(all_pts),
            'sigma_iso': round(s_iso, 5) if not np.isnan(s_iso) else None,
            'sigma_all': round(s_all, 5) if not np.isnan(s_all) else None,
            'delta': round(delta, 5) if not np.isnan(delta) else None,
        })

        si_str = f"{s_iso:8.4f}" if not np.isnan(s_iso) else "     ---"
        sa_str = f"{s_all:8.4f}" if not np.isnan(s_all) else "     ---"
        d_str = f"{delta:+8.4f}" if not np.isnan(delta) else "     ---"
        n_all_str = f"{len(all_pts):6d}" if len(all_pts) > 0 else "   ---"

        print(f"  {center:10.2f} {len(iso_pts):6d} {si_str} {n_all_str} {sa_str} {d_str}")

# ================================================================
# TEST 6: DWARFS vs MASSIVE — scatter convergence at low acceleration
# ================================================================
print(f"\n{'='*72}")
print("TEST 6: DWARFS vs MASSIVE — CONVERGENCE TEST")
print(f"{'='*72}")

if 'isolated_dwarfs' in all_residuals and 'isolated_mass4' in all_residuals:
    dwarf_res = np.array([r['residual'] for r in all_residuals['isolated_dwarfs']])
    massive_res = np.array([r['residual'] for r in all_residuals['isolated_mass4']])

    print(f"  Isolated dwarfs (M* < 10^10.3): σ = {np.std(dwarf_res):.4f}  (N={len(dwarf_res)})")
    print(f"  Isolated massive (M* > 10^10.8): σ = {np.std(massive_res):.4f}  (N={len(massive_res)})")
    print(f"  Δσ(massive-dwarf) = {np.std(massive_res) - np.std(dwarf_res):+.4f}")

    # The BEC prediction: at the LOW acceleration end of each sample,
    # scatter should converge. At the HIGH acceleration end, it may diverge.
    dwarf_gbar = np.array([r['log_gbar'] for r in all_residuals['isolated_dwarfs']])
    massive_gbar = np.array([r['log_gbar'] for r in all_residuals['isolated_mass4']])

    print(f"\n  Dwarf acceleration range:   [{dwarf_gbar.min():.1f}, {dwarf_gbar.max():.1f}]")
    print(f"  Massive acceleration range: [{massive_gbar.min():.1f}, {massive_gbar.max():.1f}]")

# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*72}")
print("OVERALL VERDICT")
print(f"{'='*72}")

# Key question: does the isolated vs all scatter difference depend on mass/acceleration?
if iso_vs_all:
    low_mass_delta = iso_vs_all[0]['delta']
    high_mass_delta = iso_vs_all[-1]['delta']

    if abs(low_mass_delta) < abs(high_mass_delta) and high_mass_delta > 0:
        lensing_verdict = ("BEC-CONSISTENT: environment (isolated vs all) matters MORE at "
                           "high mass/acceleration. Low-mass lensing RAR is environment-insensitive.")
    elif abs(low_mass_delta) > abs(high_mass_delta):
        lensing_verdict = ("OPPOSITE TO BEC: environment matters MORE at low mass.")
    else:
        lensing_verdict = ("INCONCLUSIVE: no clear mass-dependent environmental sensitivity.")
else:
    lensing_verdict = "INSUFFICIENT DATA for environment comparison."

print(f"\n  {lensing_verdict}")

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'brouwer_lensing_rar_scatter',
    'description': ('Brouwer+2021 KiDS-1000 weak lensing RAR: scatter analysis '
                    'across mass bins (acceleration proxy) and isolated vs all '
                    '(environment proxy). Tests whether lensing RAR shows same '
                    'phase transition pattern as kinematic RAR.'),
    'datasets_loaded': {name: len(data) for name, data in datasets.items() if data},
    'rar_residuals': {
        name: {
            'n_points': len(res),
            'mean_residual': round(float(np.mean([r['residual'] for r in res])), 5),
            'std_residual': round(float(np.std([r['residual'] for r in res])), 5),
        }
        for name, res in all_residuals.items()
    },
    'isolated_vs_all': iso_vs_all,
    'acceleration_profile': acc_profile if combined_iso else [],
    'verdict': lensing_verdict,
}

outpath = os.path.join(RESULTS_DIR, 'summary_brouwer_lensing_rar.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
