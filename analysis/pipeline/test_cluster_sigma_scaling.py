#!/usr/bin/env python3
"""
Test A: Cluster velocity dispersion vs acceleration scale
==========================================================

Tests BEC prediction: if g† is a condensation temperature with T ∝ σ²,
then per-cluster g‡ should scale as:

    log(g‡) = A + B × log(σ²)

with B ≈ 1.

Uses per-cluster g‡ from our E2 analysis (Tian+2020 CLASH data) and
published velocity dispersions from CLASH-VLT spectroscopy + virial
estimates from M200/r200 (Pizzardo+2025, Adam+2022).

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# CONSTANTS
# ================================================================
g_dagger = 1.20e-10          # galaxy-scale acceleration (m/s²)
sigma_galaxy = 136.0          # characteristic galaxy σ (km/s)
G_pc = 4.302e-3              # G in pc (km/s)² / M_sun

print("=" * 72)
print("TEST A: CLUSTER σ vs ACCELERATION SCALE")
print("=" * 72)

# ================================================================
# 1. LOAD PER-CLUSTER g‡ FROM E2 RESULTS
# ================================================================
print("\n[1] Loading per-cluster g‡ from E2 analysis...")

e2_path = os.path.join(RESULTS_DIR, 'summary_cluster_rar_tian2020.json')
with open(e2_path) as f:
    e2 = json.load(f)

# Build dict: name -> log_a0 (= log g‡)
cluster_gdagger = {}
for entry in e2['per_cluster_a0']:
    cluster_gdagger[entry['name']] = {
        'log_gdagger': entry['log_a0'],
        'gdagger': 10**entry['log_a0'],
        'scatter': entry['scatter'],
        'n_pts': entry['n_pts'],
    }

print(f"  Loaded g‡ for {len(cluster_gdagger)} clusters")

# ================================================================
# 2. COMPILE VELOCITY DISPERSIONS
# ================================================================
print("\n[2] Compiling cluster velocity dispersions...")

# Helper: compute σ_200 from M200 and r200 via virial theorem
# M = 3 σ² R / G  =>  σ = sqrt(G M / (3 R))
def virial_sigma(m200_msun, r200_mpc):
    """Compute 1D velocity dispersion from virial theorem.

    Args:
        m200_msun: M200 in solar masses
        r200_mpc: r200 in Mpc
    Returns:
        sigma in km/s
    """
    r200_pc = r200_mpc * 1e6  # Mpc -> pc
    return np.sqrt(G_pc * m200_msun / (3.0 * r200_pc))

# Direct spectroscopic measurements (preferred)
sigma_direct = {
    'A209':    {'sigma': 1325, 'e_sigma': 75, 'source': 'Mercurio+2003', 'method': 'spectroscopic'},
    'A2261':   {'sigma': 725,  'e_sigma': 66, 'source': 'Rines+, dynamical', 'method': 'spectroscopic',
                'flag': 'possibly biased low'},
    'MACS0329':{'sigma': 1018, 'e_sigma': 44, 'source': 'Girardi+2024 (within R200)', 'method': 'spectroscopic'},
    'MACS0416':{'sigma': 1000, 'e_sigma': 50, 'source': 'Balestra+2016', 'method': 'spectroscopic',
                'flag': 'active merger, bimodal velocity'},
    'MACS1149':{'sigma': 1840, 'e_sigma': 200,'source': 'Smith+2009 (68 redshifts)', 'method': 'spectroscopic',
                'flag': 'triple merger, likely biased high'},
    'MACS1206':{'sigma': 1087, 'e_sigma': 29, 'source': 'Biviano+2013', 'method': 'spectroscopic'},
    'RXJ1347': {'sigma': 1163, 'e_sigma': 97, 'source': 'Lu+2010', 'method': 'spectroscopic'},
    'RXJ2248': {'sigma': 1380, 'e_sigma': 29, 'source': 'Mercurio+2021', 'method': 'spectroscopic'},
}

# Virial estimates from M200/r200 (Pizzardo+2025 MAMPOSSt analysis)
pizzardo_clusters = {
    'A383':    {'m200': 8.4e14,  'r200': 1.83, 'source': 'Pizzardo+2025'},
    'MACS1115':{'m200': 10.7e14, 'r200': 1.87, 'source': 'Pizzardo+2025'},
    'MACS1931':{'m200': 11.5e14, 'r200': 1.91, 'source': 'Pizzardo+2025'},
    'MS2137':  {'m200': 7.9e14,  'r200': 1.70, 'source': 'Pizzardo+2025'},
    'RXJ2129': {'m200': 7.7e14,  'r200': 1.75, 'source': 'Pizzardo+2025'},
}

# Additional virial estimate
other_virial = {
    'MACS0647':{'m200': 18.1e14, 'r200': 2.060, 'source': 'Adam+2022'},
}

# Compute virial σ
sigma_virial = {}
for name, vals in {**pizzardo_clusters, **other_virial}.items():
    sig = virial_sigma(vals['m200'], vals['r200'])
    sigma_virial[name] = {
        'sigma': round(sig, 1),
        'e_sigma': round(sig * 0.15, 1),  # ~15% systematic uncertainty
        'source': f"{vals['source']} (virial: M200={vals['m200']:.1e}, r200={vals['r200']:.2f} Mpc)",
        'method': 'virial',
    }

# Merge: prefer direct measurements
all_sigma = {}
for name, vals in sigma_virial.items():
    if name not in sigma_direct:
        all_sigma[name] = vals
for name, vals in sigma_direct.items():
    all_sigma[name] = vals

print(f"  Direct spectroscopic σ: {len(sigma_direct)} clusters")
print(f"  Virial σ from M200/r200: {len(sigma_virial)} clusters")
print(f"  Total unique: {len(all_sigma)} clusters")

# ================================================================
# 3. MATCH CLUSTERS WITH BOTH σ AND g‡
# ================================================================
print("\n[3] Matching clusters with both σ and g‡...")

matched = []
for name in sorted(all_sigma.keys()):
    if name in cluster_gdagger:
        s = all_sigma[name]
        g = cluster_gdagger[name]
        entry = {
            'name': name,
            'sigma': s['sigma'],
            'e_sigma': s['e_sigma'],
            'sigma_method': s['method'],
            'sigma_source': s['source'],
            'log_gdagger': g['log_gdagger'],
            'gdagger': g['gdagger'],
            'rar_scatter': g['scatter'],
            'rar_n_pts': g['n_pts'],
            'flag': s.get('flag', None),
        }
        matched.append(entry)
        flag_str = f" [{s.get('flag', '')}]" if s.get('flag') else ""
        print(f"  {name:<12} σ={s['sigma']:>6.0f} km/s ({s['method']:<13}) "
              f"log g‡={g['log_gdagger']:>7.3f}{flag_str}")
    else:
        print(f"  {name:<12} σ={all_sigma[name]['sigma']:>6.0f} km/s — NO g‡ MATCH")

n_matched = len(matched)
print(f"\n  Matched: {n_matched} clusters")

# ================================================================
# 4. FIT: log(g‡) = A + B × log(σ²)
# ================================================================
print("\n" + "=" * 72)
print("TEST: log(g‡) = A + B × log(σ²)")
print("  BEC prediction: B ≈ 1")
print("=" * 72)

# Arrays for fitting
sigma_arr = np.array([m['sigma'] for m in matched])
log_gdagger_arr = np.array([m['log_gdagger'] for m in matched])
log_sigma2 = np.log10(sigma_arr**2)

# Identify unflagged (clean) subset
clean_mask = np.array([m['flag'] is None for m in matched])
n_clean = np.sum(clean_mask)

# --- Full sample ---
print(f"\n  === Full sample ({n_matched} clusters) ===")
slope, intercept, r_value, p_value, std_err = linregress(log_sigma2, log_gdagger_arr)
r_pearson, p_pearson = pearsonr(log_sigma2, log_gdagger_arr)
r_spearman, p_spearman = spearmanr(log_sigma2, log_gdagger_arr)

print(f"  B (slope)     = {slope:.4f} ± {std_err:.4f}")
print(f"  A (intercept) = {intercept:.4f}")
print(f"  Pearson  r = {r_pearson:+.4f}, p = {p_pearson:.4e}")
print(f"  Spearman ρ = {r_spearman:+.4f}, p = {p_spearman:.4e}")

if abs(slope - 1.0) < 2 * std_err:
    bec_verdict_full = "CONSISTENT with BEC prediction B≈1"
elif slope > 0 and p_pearson < 0.05:
    bec_verdict_full = f"POSITIVE correlation but B={slope:.2f} deviates from 1"
elif p_pearson >= 0.05:
    bec_verdict_full = "NO SIGNIFICANT correlation detected"
else:
    bec_verdict_full = f"UNEXPECTED: B={slope:.2f}"

print(f"  → {bec_verdict_full}")

# Residuals
pred_full = slope * log_sigma2 + intercept
resid_full = log_gdagger_arr - pred_full
rms_full = np.std(resid_full)
print(f"  RMS scatter = {rms_full:.4f} dex")

# --- Clean sample (no flags) ---
print(f"\n  === Clean sample ({n_clean} clusters, no flags) ===")
slope_c, intercept_c, r_c, p_c, std_err_c = linregress(
    log_sigma2[clean_mask], log_gdagger_arr[clean_mask])
r_p_c, p_p_c = pearsonr(log_sigma2[clean_mask], log_gdagger_arr[clean_mask])
r_s_c, p_s_c = spearmanr(log_sigma2[clean_mask], log_gdagger_arr[clean_mask])

print(f"  B (slope)     = {slope_c:.4f} ± {std_err_c:.4f}")
print(f"  A (intercept) = {intercept_c:.4f}")
print(f"  Pearson  r = {r_p_c:+.4f}, p = {p_p_c:.4e}")
print(f"  Spearman ρ = {r_s_c:+.4f}, p = {p_s_c:.4e}")

if abs(slope_c - 1.0) < 2 * std_err_c:
    bec_verdict_clean = "CONSISTENT with BEC prediction B≈1"
elif slope_c > 0 and p_p_c < 0.05:
    bec_verdict_clean = f"POSITIVE correlation but B={slope_c:.2f} deviates from 1"
elif p_p_c >= 0.05:
    bec_verdict_clean = "NO SIGNIFICANT correlation detected"
else:
    bec_verdict_clean = f"UNEXPECTED: B={slope_c:.2f}"

print(f"  → {bec_verdict_clean}")

# ================================================================
# 5. SCALING RATIO TEST: g‡/g† vs (σ/σ_galaxy)²
# ================================================================
print("\n" + "=" * 72)
print(f"SCALING: g‡/g† vs (σ/{sigma_galaxy:.0f})²")
print("=" * 72)

ratio_gdagger = np.array([m['gdagger'] for m in matched]) / g_dagger
ratio_sigma2 = (sigma_arr / sigma_galaxy)**2

print(f"\n  {'Cluster':<12} {'σ':>6} {'(σ/σ_gal)²':>10} {'g‡/g†':>8} {'ratio':>8}")
print(f"  " + "-" * 50)
for i, m in enumerate(matched):
    rat = ratio_gdagger[i] / ratio_sigma2[i]
    print(f"  {m['name']:<12} {m['sigma']:>6.0f} {ratio_sigma2[i]:>10.1f} "
          f"{ratio_gdagger[i]:>8.1f} {rat:>8.2f}")

# If BEC: g‡/g† = (σ/σ_galaxy)², so the ratio g‡/(g† × (σ/σ_gal)²) should be ~1
scaling_ratio = ratio_gdagger / ratio_sigma2
print(f"\n  Mean g‡ / [g† × (σ/σ_gal)²] = {np.mean(scaling_ratio):.3f}")
print(f"  Median                        = {np.median(scaling_ratio):.3f}")
print(f"  Std                           = {np.std(scaling_ratio):.3f}")

# ================================================================
# 6. BOOTSTRAP UNCERTAINTY ON SLOPE
# ================================================================
print("\n[6] Bootstrap uncertainty on slope B...")
np.random.seed(42)
n_boot = 10000
slopes_boot = np.zeros(n_boot)
for i in range(n_boot):
    idx = np.random.choice(n_matched, n_matched, replace=True)
    s_b, _, _, _, _ = linregress(log_sigma2[idx], log_gdagger_arr[idx])
    slopes_boot[i] = s_b

boot_mean = np.mean(slopes_boot)
boot_std = np.std(slopes_boot)
boot_ci = np.percentile(slopes_boot, [2.5, 97.5])

print(f"  Bootstrap B = {boot_mean:.4f} ± {boot_std:.4f}")
print(f"  95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"  B=1 within 95% CI? {'YES' if boot_ci[0] <= 1.0 <= boot_ci[1] else 'NO'}")

# ================================================================
# 7. PLOT
# ================================================================
print("\n[7] Generating plot...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: log(g‡) vs log(σ²) ---
    ax1 = axes[0]

    # Plot clean vs flagged
    for i, m in enumerate(matched):
        color = 'royalblue' if m['flag'] is None else 'orange'
        marker = 'o' if m['sigma_method'] == 'spectroscopic' else 's'
        ax1.errorbar(log_sigma2[i], m['log_gdagger'],
                     xerr=2 * m['e_sigma'] / (m['sigma'] * np.log(10)),
                     fmt=marker, color=color, markersize=8, capsize=3, zorder=5)
        ax1.annotate(m['name'], (log_sigma2[i], m['log_gdagger']),
                     fontsize=6, ha='left', va='bottom',
                     xytext=(3, 3), textcoords='offset points')

    # Best fit line
    x_fit = np.linspace(log_sigma2.min() - 0.1, log_sigma2.max() + 0.1, 50)
    ax1.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2,
             label=f'Best fit: B={slope:.3f}±{std_err:.3f}')

    # BEC prediction line (B=1)
    # Find intercept for B=1 that passes through the median point
    med_x = np.median(log_sigma2)
    med_y = np.median(log_gdagger_arr)
    bec_intercept = med_y - 1.0 * med_x
    ax1.plot(x_fit, 1.0 * x_fit + bec_intercept, 'g--', linewidth=1.5, alpha=0.7,
             label=f'BEC prediction: B=1')

    ax1.set_xlabel(r'$\log(\sigma^2 / {\rm km^2\,s^{-2}})$', fontsize=12)
    ax1.set_ylabel(r'$\log(g^\ddagger / {\rm m\,s^{-2}})$', fontsize=12)
    ax1.set_title(f'Cluster g‡ vs velocity dispersion (N={n_matched})', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add Pearson annotation
    ax1.text(0.05, 0.05, f'Pearson r={r_pearson:.3f} (p={p_pearson:.3e})\n'
             f'Spearman ρ={r_spearman:.3f} (p={p_spearman:.3e})',
             transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: g‡/g† vs (σ/σ_galaxy)² ---
    ax2 = axes[1]

    for i, m in enumerate(matched):
        color = 'royalblue' if m['flag'] is None else 'orange'
        marker = 'o' if m['sigma_method'] == 'spectroscopic' else 's'
        ax2.scatter(ratio_sigma2[i], ratio_gdagger[i],
                    c=color, marker=marker, s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax2.annotate(m['name'], (ratio_sigma2[i], ratio_gdagger[i]),
                     fontsize=6, ha='left', va='bottom',
                     xytext=(3, 3), textcoords='offset points')

    # BEC prediction: g‡/g† = (σ/σ_gal)²
    x_bec = np.linspace(0, max(ratio_sigma2) * 1.2, 50)
    ax2.plot(x_bec, x_bec, 'g--', linewidth=2, alpha=0.7, label=r'BEC: $g^\ddagger/g^\dagger = (\sigma/\sigma_{\rm gal})^2$')

    # 1:1 reference
    ax2.set_xlabel(r'$(\sigma / \sigma_{\rm galaxy})^2$', fontsize=12)
    ax2.set_ylabel(r'$g^\ddagger / g^\dagger$', fontsize=12)
    ax2.set_title(f'Acceleration scaling (σ_galaxy = {sigma_galaxy:.0f} km/s)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=8, label='Spectroscopic σ (clean)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue',
               markersize=8, label='Virial σ from M200/r200'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label='Flagged (merger)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, 'cluster_sigma_vs_gdagger.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

except ImportError:
    print("  matplotlib not available, skipping plot")
    fig_path = None

# ================================================================
# 8. SUMMARY & SAVE
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

# Determine overall verdict
if p_pearson < 0.05 and abs(slope - 1.0) < 2 * std_err:
    verdict = "BEC_PREDICTION_CONFIRMED"
    verdict_text = (f"Significant positive correlation (p={p_pearson:.3e}) with "
                    f"B={slope:.3f}±{std_err:.3f}, consistent with BEC prediction B=1")
elif p_pearson < 0.05 and slope > 0:
    verdict = "POSITIVE_CORRELATION"
    verdict_text = (f"Significant positive correlation (p={p_pearson:.3e}) with "
                    f"B={slope:.3f}±{std_err:.3f}, but deviates from BEC prediction B=1")
elif p_pearson >= 0.05:
    verdict = "NO_SIGNIFICANT_CORRELATION"
    verdict_text = f"No significant correlation (p={p_pearson:.3e}), B={slope:.3f}±{std_err:.3f}"
else:
    verdict = "UNEXPECTED"
    verdict_text = f"Unexpected result: B={slope:.3f}±{std_err:.3f}"

print(f"\n  {verdict_text}")
print(f"\n  Full sample (N={n_matched}):")
print(f"    B = {slope:.4f} ± {std_err:.4f}")
print(f"    Pearson r = {r_pearson:.4f}, p = {p_pearson:.4e}")
print(f"  Clean sample (N={n_clean}):")
print(f"    B = {slope_c:.4f} ± {std_err_c:.4f}")
print(f"    Pearson r = {r_p_c:.4f}, p = {p_p_c:.4e}")
print(f"  Bootstrap (10000 iterations):")
print(f"    B = {boot_mean:.4f} ± {boot_std:.4f}")
print(f"    95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"    B=1 in CI: {'YES' if boot_ci[0] <= 1.0 <= boot_ci[1] else 'NO'}")

results = {
    'test_name': 'cluster_sigma_scaling',
    'description': 'Test A: Does per-cluster g‡ scale with velocity dispersion as g‡ ∝ σ²?',
    'bec_prediction': 'B ≈ 1 in log(g‡) = A + B × log(σ²)',
    'n_clusters_matched': n_matched,
    'n_clusters_clean': int(n_clean),
    'sigma_galaxy_kms': sigma_galaxy,
    'g_dagger_galaxy': g_dagger,
    'full_sample': {
        'B_slope': float(slope),
        'B_stderr': float(std_err),
        'A_intercept': float(intercept),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_rho': float(r_spearman),
        'spearman_p': float(p_spearman),
        'rms_scatter': float(rms_full),
    },
    'clean_sample': {
        'B_slope': float(slope_c),
        'B_stderr': float(std_err_c),
        'A_intercept': float(intercept_c),
        'pearson_r': float(r_p_c),
        'pearson_p': float(p_p_c),
        'spearman_rho': float(r_s_c),
        'spearman_p': float(p_s_c),
    },
    'bootstrap': {
        'n_iterations': n_boot,
        'B_mean': float(boot_mean),
        'B_std': float(boot_std),
        'B_95ci': [float(boot_ci[0]), float(boot_ci[1])],
        'B1_in_ci': bool(boot_ci[0] <= 1.0 <= boot_ci[1]),
    },
    'scaling_ratio': {
        'description': 'g‡ / [g† × (σ/σ_gal)²] — should be ~1 if BEC',
        'mean': float(np.mean(scaling_ratio)),
        'median': float(np.median(scaling_ratio)),
        'std': float(np.std(scaling_ratio)),
    },
    'clusters': [
        {
            'name': m['name'],
            'sigma_kms': m['sigma'],
            'e_sigma': m['e_sigma'],
            'sigma_method': m['sigma_method'],
            'sigma_source': m['sigma_source'],
            'log_gdagger': m['log_gdagger'],
            'gdagger_over_gdagger_gal': float(m['gdagger'] / g_dagger),
            'sigma_over_sigma_gal_sq': float((m['sigma'] / sigma_galaxy)**2),
            'flag': m['flag'],
        }
        for m in matched
    ],
    'missing_clusters': [
        name for name in cluster_gdagger if name not in all_sigma
    ],
    'verdict': verdict,
    'verdict_text': verdict_text,
    'data_sources': {
        'gdagger': 'Per-cluster RAR fits from Tian+2020 CLASH data (our E2 analysis)',
        'sigma_spectroscopic': [
            'Mercurio+2003 (A209)',
            'Girardi+2024 (MACS0329)',
            'Balestra+2016 (MACS0416)',
            'Biviano+2013 (MACS1206)',
            'Lu+2010 (RXJ1347)',
            'Mercurio+2021 (RXJ2248/AS1063)',
            'Smith+2009 (MACS1149, flagged: triple merger)',
            'Rines+ (A2261, flagged: possibly biased low)',
        ],
        'sigma_virial': [
            'Pizzardo+2025 MAMPOSSt (A383, MACS1115, MACS1931, MS2137, RXJ2129)',
            'Adam+2022 (MACS0647)',
        ],
    },
}

out_path = os.path.join(RESULTS_DIR, 'summary_cluster_sigma_scaling.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")

print("=" * 72)
print("Done.")
