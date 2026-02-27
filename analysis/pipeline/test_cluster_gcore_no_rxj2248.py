#!/usr/bin/env python3
"""
Test A3b: g_core analysis with RXJ2248 excluded (anomalous Rad=30.3 kpc)
+ standardized-aperture variant (Rad=14.3 kpc subsample)
+ multivariate re-check without RXJ2248

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os, json, numpy as np
from scipy.stats import pearsonr, spearmanr, linregress, t as t_dist
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cluster_rar')

G_SI = 6.674e-11; M_sun = 1.989e30; kpc_m = 3.086e19; Mpc_m = 3.086e22
g_dagger_gal = 1.20e-10

# --- Load data (same as A3) ---
with open(os.path.join(RESULTS_DIR, 'summary_cluster_rar_tian2020.json')) as f:
    e2 = json.load(f)
cluster_gd = {e['name']: {'log_gd': e['log_a0'], 'gd': 10**e['log_a0'],
              'scatter': e['scatter'], 'n_pts': e['n_pts']} for e in e2['per_cluster_a0']}

table1_path = os.path.join(DATA_DIR, 'tian2020_table1.dat')
bcg = {}
with open(table1_path) as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('-') or 'Name' in s: continue
        p = s.split('|')
        if len(p) < 14: continue
        try:
            nm = p[-1].strip()
            if not nm or not nm[0].isalpha(): continue
            bcg[nm] = {
                'z': float(p[1].strip()),
                'Re': float(p[5].strip()) if p[5].strip() else None,
                'Rad': float(p[7].strip()) if p[7].strip() else None,
                'Mstar': float(p[-6].strip()) if p[-6].strip() else None,
                'Mgas': float(p[-5].strip()) if p[-5].strip() else None,
                'Mtot': float(p[-3].strip()) if p[-3].strip() else None,
            }
        except: continue

m200_data = {
    'A209':    {'m200': 17.3e14, 'r200': 2.31, 'c200': 3.4},
    'A383':    {'m200': 8.4e14,  'r200': 1.83, 'c200': 2.5},
    'MACS0329':{'m200': 11.5e14, 'r200': 1.84, 'c200': 5.4},
    'MACS1115':{'m200': 10.7e14, 'r200': 1.87, 'c200': 2.5},
    'MACS1206':{'m200': 15.9e14, 'r200': 2.06, 'c200': 5.8},
    'MACS1931':{'m200': 11.5e14, 'r200': 1.91, 'c200': 7.8},
    'MS2137':  {'m200': 7.9e14,  'r200': 1.70, 'c200': 2.4},
    'RXJ2129': {'m200': 7.7e14,  'r200': 1.75, 'c200': 2.9},
    'RXJ2248': {'m200': 22.7e14, 'r200': 2.40, 'c200': 1.6},
    'MACS0647':{'m200': 18.1e14, 'r200': 2.06, 'c200': None},
}

def compute_gcore(mtot_1e12, rad_kpc):
    return G_SI * (mtot_1e12 * 1e12 * M_sun) / (rad_kpc * kpc_m)**2

def compute_g200(m200, r200_mpc):
    return G_SI * (m200 * M_sun) / (r200_mpc * Mpc_m)**2

def ols(X, y):
    """OLS with coefficients, SE, t, p."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    n, k = X.shape; dof = n - k
    mse = np.sum(resid**2) / dof if dof > 0 else np.inf
    try:
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except: se = np.full(k, np.nan)
    t_vals = b / se
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof)) if dof > 0 else np.full(k, 1.0)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else 0
    return b, se, t_vals, p_vals, r2, r2_adj, np.std(resid)

print("=" * 72)
print("TEST A3b: g_core WITHOUT RXJ2248 + STANDARDIZED APERTURES")
print("=" * 72)

# ================================================================
# Build master table
# ================================================================
rows = []
for nm in sorted(bcg.keys()):
    if nm not in cluster_gd or bcg[nm]['Mtot'] is None or bcg[nm]['Rad'] is None:
        continue
    b = bcg[nm]
    gc = compute_gcore(b['Mtot'], b['Rad'])
    row = {'name': nm, 'Rad': b['Rad'], 'Mtot': b['Mtot'], 'Mstar': b['Mstar'],
           'z': b['z'], 'log_gcore': np.log10(gc), 'gcore': gc,
           'log_gd': cluster_gd[nm]['log_gd'], 'gd': cluster_gd[nm]['gd']}
    if nm in m200_data:
        d = m200_data[nm]
        row['m200'] = d['m200']; row['r200'] = d['r200']; row['c200'] = d['c200']
        row['g200'] = compute_g200(d['m200'], d['r200'])
        row['log_g200'] = np.log10(row['g200'])
    rows.append(row)

# ================================================================
# 1. UNIVARIATE: g_core vs g‡ — three variants
# ================================================================
print("\n" + "=" * 72)
print("1. UNIVARIATE: log g‡ vs log g_core")
print("=" * 72)

def run_univariate(subset, label):
    x = np.array([r['log_gcore'] for r in subset])
    y = np.array([r['log_gd'] for r in subset])
    sl, ic, _, _, se = linregress(x, y)
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    # Bootstrap B
    np.random.seed(42)
    B_boot = [linregress(x[idx:=np.random.choice(len(x),len(x),replace=True)],
                          y[idx])[0] for _ in range(10000)]
    ci = np.percentile(B_boot, [2.5, 97.5])
    ratio = np.array([r['gd']/r['gcore'] for r in subset])

    print(f"\n  --- {label} (N={len(subset)}) ---")
    print(f"  Rad values: {sorted(set(r['Rad'] for r in subset))}")
    print(f"  B = {sl:.4f} ± {se:.4f}  [bootstrap 95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Pearson  r = {rp:+.4f}, p = {pp:.4e}")
    print(f"  Spearman ρ = {rs:+.4f}, p = {ps:.4e}")
    print(f"  B=1 in CI? {'YES' if ci[0]<=1<=ci[1] else 'NO'}  |  B=0 in CI? {'YES' if ci[0]<=0<=ci[1] else 'NO'}")
    print(f"  g‡/g_core: median={np.median(ratio):.3f}, mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}")
    sig = '***' if pp<0.01 else '**' if pp<0.05 else '*' if pp<0.1 else ''
    print(f"  {sig}")
    return {'label': label, 'n': len(subset), 'B': sl, 'B_err': se,
            'r': rp, 'p': pp, 'rho': rs, 'rho_p': ps, 'ci': ci.tolist(),
            'ratio_med': float(np.median(ratio)), 'ratio_std': float(np.std(ratio))}

# (a) All 20
res_all = run_univariate(rows, "All 20 clusters")

# (b) Drop RXJ2248
rows_no2248 = [r for r in rows if r['name'] != 'RXJ2248']
res_no2248 = run_univariate(rows_no2248, "Excluding RXJ2248 (N=19)")

# (c) Rad=14.3 kpc subsample only (standardized aperture)
rows_14 = [r for r in rows if r['Rad'] == 14.3]
res_14 = run_univariate(rows_14, "Rad=14.3 kpc only (standardized)")

# (d) Drop ALL non-14.3 clusters (stricter)
non_14 = [r['name'] for r in rows if r['Rad'] != 14.3]
print(f"\n  Non-14.3 kpc clusters excluded: {non_14}")

# ================================================================
# 2. MULTIVARIATE: with and without RXJ2248
# ================================================================
print("\n" + "=" * 72)
print("2. MULTIVARIATE REGRESSION")
print("=" * 72)

def run_multivariate(subset, label):
    # Need g_core + c200 + g200
    mv = [r for r in subset if 'c200' in r and r['c200'] is not None]
    if len(mv) < 5:
        print(f"\n  --- {label}: only {len(mv)} clusters, skipping ---")
        return None

    names = [r['name'] for r in mv]
    y = np.array([r['log_gd'] for r in mv])
    x_gc = np.array([r['log_gcore'] for r in mv])
    x_c = np.array([np.log10(r['c200']) for r in mv])
    x_g2 = np.array([r['log_g200'] for r in mv])

    print(f"\n  --- {label} (N={len(mv)}): {', '.join(names)} ---")

    # Full model: g_core + c200 + g200
    X_full = np.column_stack([np.ones(len(mv)), x_gc, x_c, x_g2])
    labs_full = ['intercept', 'log g_core', 'log c_200', 'log g_200']
    b, se, t, p, r2, r2a, rms = ols(X_full, y)

    print(f"\n  Full model: log g‡ = a + b×log(g_core) + c×log(c200) + d×log(g200)")
    print(f"  R² = {r2:.4f}, Adj R² = {r2a:.4f}, RMS = {rms:.4f} dex")
    print(f"  {'Predictor':<18} {'Coeff':>8} {'SE':>8} {'t':>8} {'p':>10} {'':>5}")
    print(f"  " + "-" * 55)
    for i, lab in enumerate(labs_full):
        sig = '***' if p[i]<0.01 else '**' if p[i]<0.05 else '*' if p[i]<0.1 else ''
        print(f"  {lab:<18} {b[i]:>+8.4f} {se[i]:>8.4f} {t[i]:>+8.3f} {p[i]:>10.4e} {sig}")

    # Bivariate: g_core + g200 only
    X_bg = np.column_stack([np.ones(len(mv)), x_gc, x_g2])
    b2, se2, t2, p2, r2_bg, r2a_bg, rms2 = ols(X_bg, y)
    print(f"\n  Bivariate g_core + g200:")
    print(f"  R² = {r2_bg:.4f}, Adj R² = {r2a_bg:.4f}")
    for i, lab in enumerate(['intercept', 'log g_core', 'log g_200']):
        sig = '***' if p2[i]<0.01 else '**' if p2[i]<0.05 else '*' if p2[i]<0.1 else ''
        print(f"    {lab:<18} {b2[i]:>+8.4f} ± {se2[i]:.4f}  p={p2[i]:.4e} {sig}")

    # Univariate: g_core only
    X_gc = np.column_stack([np.ones(len(mv)), x_gc])
    _, _, _, p_gc, r2_gc, r2a_gc, _ = ols(X_gc, y)
    print(f"\n  Univariate g_core only: R² = {r2_gc:.4f}, p = {p_gc[1]:.4e}")

    # Univariate: g200 only
    X_g2o = np.column_stack([np.ones(len(mv)), x_g2])
    _, _, _, p_g2, r2_g2, r2a_g2, _ = ols(X_g2o, y)
    print(f"  Univariate g200 only:  R² = {r2_g2:.4f}, p = {p_g2[1]:.4e}")

    return {
        'label': label, 'n': len(mv), 'names': names,
        'full_R2': r2, 'full_R2_adj': r2a,
        'full_coeff': {labs_full[i]: {'b': float(b[i]), 'se': float(se[i]),
                       'p': float(p[i])} for i in range(len(labs_full))},
        'bivar_gc_g200_R2': r2_bg, 'bivar_gc_g200_R2_adj': r2a_bg,
        'univar_gcore_R2': r2_gc, 'univar_g200_R2': r2_g2,
    }

mv_all = run_multivariate(rows, "All clusters with g_core + c200 + g200")
mv_no2248 = run_multivariate(rows_no2248, "Excluding RXJ2248")

# Also test: g_core only on the multivariate subset (with vs without RXJ2248)
print(f"\n  --- g_core univariate on multivariate subsets ---")
for label, subset in [("With RXJ2248", rows), ("Without RXJ2248", rows_no2248)]:
    mv = [r for r in subset if 'c200' in r and r['c200'] is not None]
    x = np.array([r['log_gcore'] for r in mv])
    y = np.array([r['log_gd'] for r in mv])
    sl, ic, _, _, se = linregress(x, y)
    rp, pp = pearsonr(x, y)
    print(f"  {label} (N={len(mv)}): B={sl:.4f}±{se:.4f}, r={rp:+.4f}, p={pp:.4e}")

# ================================================================
# 3. SUMMARY TABLE
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY: UNIVARIATE g_core → g‡ COMPARISON")
print("=" * 72)

print(f"\n  {'Sample':<35} {'N':>3} {'B':>7} {'SE':>6} {'r':>7} {'p':>10} {'g‡/gc med':>9}")
print(f"  " + "-" * 80)
for res in [res_all, res_no2248, res_14]:
    sig = '***' if res['p']<0.01 else '**' if res['p']<0.05 else '*' if res['p']<0.1 else ''
    print(f"  {res['label']:<35} {res['n']:>3} {res['B']:>+7.4f} {res['B_err']:>6.4f} "
          f"{res['r']:>+7.4f} {res['p']:>10.4e} {res['ratio_med']:>9.3f} {sig}")

# ================================================================
# PLOT
# ================================================================
print("\n[PLOT]")
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    # Helper
    def scatter_fit(ax, rows_sub, title, exclude_names=None):
        x = np.array([r['log_gcore'] for r in rows_sub])
        y = np.array([r['log_gd'] for r in rows_sub])
        sl, ic, _, _, se = linregress(x, y)
        rp, pp = pearsonr(x, y)

        for r in rows_sub:
            c = 'red' if r['name'] == 'RXJ2248' else ('orange' if r['Rad'] != 14.3 else 'royalblue')
            ax.scatter(r['log_gcore'], r['log_gd'], c=c, s=55, zorder=5,
                       edgecolors='k', linewidth=0.5)
            ax.annotate(r['name'], (r['log_gcore'], r['log_gd']), fontsize=5.5,
                        xytext=(3, 3), textcoords='offset points')

        xf = np.linspace(x.min()-0.03, x.max()+0.03, 50)
        ax.plot(xf, sl*xf + ic, 'r-', lw=2,
                label=f'B={sl:.3f}±{se:.3f}\nr={rp:+.3f}, p={pp:.2e}')
        mx, my = np.median(x), np.median(y)
        ax.plot(xf, 1.0*(xf-mx)+my, 'g--', lw=1.2, alpha=0.5, label='B=1')
        ax.set_xlabel(r'$\log(g_{\rm core})$ [m/s²]')
        ax.set_ylabel(r'$\log(g^\ddagger)$ [m/s²]')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Panel 1: All 20
    scatter_fit(axes[0,0], rows, f'All 20 clusters')

    # Panel 2: No RXJ2248
    scatter_fit(axes[0,1], rows_no2248, f'Excluding RXJ2248 (N=19)')

    # Panel 3: Rad=14.3 only
    scatter_fit(axes[0,2], rows_14, f'Rad=14.3 kpc only (N={len(rows_14)})')

    # Panel 4: Multivariate actual vs predicted (all 9)
    mv_all_sub = [r for r in rows if 'c200' in r and r['c200'] is not None]
    if len(mv_all_sub) >= 5:
        y_mv = np.array([r['log_gd'] for r in mv_all_sub])
        X_mv = np.column_stack([np.ones(len(mv_all_sub)),
                                [r['log_gcore'] for r in mv_all_sub],
                                [np.log10(r['c200']) for r in mv_all_sub],
                                [r['log_g200'] for r in mv_all_sub]])
        b_mv = np.linalg.lstsq(X_mv, y_mv, rcond=None)[0]
        y_pred = X_mv @ b_mv
        ax4 = axes[1,0]
        for i, r in enumerate(mv_all_sub):
            c = 'red' if r['name']=='RXJ2248' else 'royalblue'
            ax4.scatter(y_pred[i], y_mv[i], c=c, s=55, zorder=5, edgecolors='k', linewidth=0.5)
            ax4.annotate(r['name'], (y_pred[i], y_mv[i]), fontsize=6,
                         xytext=(3,3), textcoords='offset points')
        lims = [min(y_pred.min(), y_mv.min())-0.02, max(y_pred.max(), y_mv.max())+0.02]
        ax4.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ss = np.sum((y_mv-y_pred)**2); st = np.sum((y_mv-np.mean(y_mv))**2)
        ax4.set_title(f'Multivariate ALL (R²={1-ss/st:.3f}, N={len(mv_all_sub)})', fontsize=10)
        ax4.set_xlabel('Predicted log g‡'); ax4.set_ylabel('Observed log g‡')
        ax4.grid(True, alpha=0.3)

    # Panel 5: Multivariate without RXJ2248
    mv_no = [r for r in rows_no2248 if 'c200' in r and r['c200'] is not None]
    if len(mv_no) >= 5:
        y_mv2 = np.array([r['log_gd'] for r in mv_no])
        X_mv2 = np.column_stack([np.ones(len(mv_no)),
                                 [r['log_gcore'] for r in mv_no],
                                 [np.log10(r['c200']) for r in mv_no],
                                 [r['log_g200'] for r in mv_no]])
        b_mv2 = np.linalg.lstsq(X_mv2, y_mv2, rcond=None)[0]
        y_pred2 = X_mv2 @ b_mv2
        ax5 = axes[1,1]
        for i, r in enumerate(mv_no):
            ax5.scatter(y_pred2[i], y_mv2[i], c='royalblue', s=55, zorder=5,
                        edgecolors='k', linewidth=0.5)
            ax5.annotate(r['name'], (y_pred2[i], y_mv2[i]), fontsize=6,
                         xytext=(3,3), textcoords='offset points')
        lims2 = [min(y_pred2.min(), y_mv2.min())-0.02, max(y_pred2.max(), y_mv2.max())+0.02]
        ax5.plot(lims2, lims2, 'k--', lw=1, alpha=0.5)
        ss2 = np.sum((y_mv2-y_pred2)**2); st2 = np.sum((y_mv2-np.mean(y_mv2))**2)
        ax5.set_title(f'Multivariate NO RXJ2248 (R²={1-ss2/st2:.3f}, N={len(mv_no)})', fontsize=10)
        ax5.set_xlabel('Predicted log g‡'); ax5.set_ylabel('Observed log g‡')
        ax5.grid(True, alpha=0.3)

    # Panel 6: g‡/g_core ratio histogram
    ax6 = axes[1,2]
    ratio_all = [r['gd']/r['gcore'] for r in rows]
    ratio_no = [r['gd']/r['gcore'] for r in rows_no2248]
    ratio_14 = [r['gd']/r['gcore'] for r in rows_14]
    ax6.hist(ratio_all, bins=10, alpha=0.3, color='gray', label=f'All 20 (med={np.median(ratio_all):.2f})')
    ax6.hist(ratio_no, bins=10, alpha=0.5, color='royalblue', label=f'No RXJ2248 (med={np.median(ratio_no):.2f})')
    ax6.axvline(np.median(ratio_no), color='royalblue', ls='--')
    ax6.set_xlabel(r'$g^\ddagger / g_{\rm core}$')
    ax6.set_ylabel('Count')
    ax6.set_title('g‡ / g_core ratio distribution')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='royalblue', markersize=8, label='Rad=14.3 kpc'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Rad≠14.3 kpc'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=8, label='RXJ2248 (Rad=30.3)'),
    ]
    axes[0,0].legend(handles=legend_els + axes[0,0].get_legend().legend_handles,
                     fontsize=6.5, loc='lower right')

    plt.suptitle('Test A3b: g_core → g‡ with aperture standardization', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'cluster_gcore_standardized.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
except ImportError:
    fig_path = None

# ================================================================
# SAVE
# ================================================================
summary = {
    'test_name': 'cluster_gcore_standardized_A3b',
    'univariate': {
        'all_20': res_all,
        'no_RXJ2248': res_no2248,
        'rad_14_3_only': res_14,
    },
    'multivariate_all': mv_all,
    'multivariate_no_RXJ2248': mv_no2248,
}
out_path = os.path.join(RESULTS_DIR, 'summary_cluster_gcore_standardized.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved: {out_path}")
print("=" * 72)
