#!/usr/bin/env python3
"""
Spectral Slope α Test (Step 30)
================================

Estimate per-galaxy power-spectrum slope α from Lomb–Scargle periodogram
of detrended RAR residuals vs radius.

For each galaxy (N>=20):
  1) LS power P(k) on log-spaced frequency grid
  2) Fit log P = -α log k + C via Theil–Sen robust regression
  3) Per-galaxy fit band: k_min = 2π/(0.5*R_ext), k_max = 2π/(2*med_dR)
  4) Surrogate null: shuffle residuals, refit α, M=200 per galaxy

Aggregate: α distribution, detrending stability, confound correlations.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os, sys, json, csv, time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle
from multiprocessing import Pool, cpu_count
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

g_dagger_SI  = 1.20e-10
kpc_m        = 3.086e19
MIN_POINTS   = 20
N_FREQ       = 200        # log-spaced frequency samples
MIN_BAND_PTS = 10         # minimum freq samples in fit band
N_SURR       = 200        # surrogates per galaxy
DETREND_MULTS = [0.5, 1.0, 2.0]
PRIMARY_MULT  = 0.5
N_WORKERS     = max(1, cpu_count() - 1)

np.random.seed(42)

print("=" * 72)
print("SPECTRAL SLOPE α TEST (Step 30)")
print("=" * 72)
print(f"  Min points: {MIN_POINTS}, Freq grid: {N_FREQ} log-spaced")
print(f"  Fit band: k_min=2π/(0.5*R_ext), k_max=2π/(2*med_dR)")
print(f"  Surrogates: {N_SURR}/galaxy, Workers: {N_WORKERS}")
print(f"  Detrending sweep: s = n*var * {DETREND_MULTS}")


# ---------- helpers ----------
def rar(lg):
    gb = 10.**lg
    return np.log10(gb / (1. - np.exp(-np.sqrt(gb / 1.2e-10))))

def detrend_residuals(Rs, eps, mult):
    """Spline-detrend residuals with given smoothing multiplier."""
    n = len(eps)
    v = np.var(eps)
    s = n * v * mult
    try:
        sp = UnivariateSpline(Rs, eps, k=min(3, n-1), s=s)
        return eps - sp(Rs)
    except:
        return eps - np.mean(eps)

def fit_alpha(R, eps_det, R_ext, med_dR):
    """
    Fit spectral slope α from LS periodogram.
    Returns (alpha, n_band, r2_approx, resid_scatter) or Nones.
    """
    n = len(eps_det)
    std_eps = np.std(eps_det)
    if std_eps < 1e-30:
        return None, 0, None, None

    y = (eps_det - np.mean(eps_det)) / std_eps

    # Fit band boundaries (in cycles/kpc, i.e. k = 1/λ, NOT angular)
    # k_min corresponds to longest wavelength = 0.5 * R_ext
    # k_max corresponds to shortest wavelength = 2 * med_dR
    if R_ext <= 0 or med_dR <= 0:
        return None, 0, None, None
    k_min = 1.0 / (0.5 * R_ext)
    k_max = 1.0 / (2.0 * med_dR)
    if k_max <= k_min:
        return None, 0, None, None

    # Log-spaced frequency grid spanning full range, then cut to band
    # Use wider grid for LS, then select band for fitting
    k_grid_full = np.logspace(np.log10(k_min * 0.5), np.log10(k_max * 2.0), N_FREQ)

    # LS periodogram (frequency = k in cycles/kpc)
    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(k_grid_full)

    # Select in-band points
    in_band = (k_grid_full >= k_min) & (k_grid_full <= k_max) & (power > 0)
    n_band = int(np.sum(in_band))
    if n_band < MIN_BAND_PTS:
        return None, n_band, None, None

    log_k = np.log10(k_grid_full[in_band])
    log_P = np.log10(power[in_band])

    # Theil-Sen robust regression: log P = -α * log k + C
    # scipy's theilslopes: slope, intercept, lo_slope, hi_slope
    try:
        slope, intercept, lo_slope, hi_slope = stats.theilslopes(log_P, log_k)
    except:
        return None, n_band, None, None

    alpha = -slope  # convention: P ∝ k^{-α}, so slope of log P vs log k is -α

    # Approximate R² from residuals
    predicted = slope * log_k + intercept
    ss_res = np.sum((log_P - predicted)**2)
    ss_tot = np.sum((log_P - np.mean(log_P))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_scatter = np.sqrt(ss_res / n_band) if n_band > 0 else np.nan

    return float(alpha), n_band, float(r2), float(resid_scatter)


def process_galaxy_surrogates(args):
    """Worker function for parallel surrogate generation."""
    R, eps_det, R_ext, med_dR, n_surr, seed = args
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(n_surr):
        eps_shuf = rng.permutation(eps_det)
        a, nb, _, _ = fit_alpha(R, eps_shuf, R_ext, med_dR)
        if a is not None:
            alphas.append(a)
    return alphas


# ================================================================
# 1. LOAD SPARC DATA
# ================================================================
print("\n[1] Loading SPARC data...")

t2  = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

rc = {}
with open(t2) as f:
    for ln in f:
        if len(ln.strip()) < 50: continue
        try:
            nm = ln[0:11].strip()
            if not nm: continue
            r = float(ln[19:25]); vo = float(ln[26:32])
            vg = float(ln[39:45]); vd = float(ln[46:52]); vb = float(ln[53:59])
        except: continue
        if nm not in rc:
            rc[nm] = {'R':[],'Vo':[],'Vg':[],'Vd':[],'Vb':[]}
        rc[nm]['R'].append(r); rc[nm]['Vo'].append(vo)
        rc[nm]['Vg'].append(vg); rc[nm]['Vd'].append(vd); rc[nm]['Vb'].append(vb)
for nm in rc:
    for k in ['R','Vo','Vg','Vd','Vb']: rc[nm][k] = np.array(rc[nm][k])

props = {}
with open(mrt) as f: lines = f.readlines()
ds = 0
for i, ln in enumerate(lines):
    if ln.startswith('---') and i > 50: ds = i + 1; break
for ln in lines[ds:]:
    if not ln.strip(): continue
    try:
        nm = ln[0:11].strip(); p = ln[11:].split()
        if len(p) < 17: continue
        props[nm] = {
            'D': float(p[1]), 'eD': float(p[2]), 'Inc': float(p[4]),
            'L36': float(p[6]), 'MHI': float(p[12]),
            'Vflat': float(p[14]), 'Q': int(p[16]),
        }
    except: continue


# ================================================================
# 2. BUILD PER-GALAXY DATA
# ================================================================
print("\n[2] Building per-galaxy residuals...")

galaxies = []
for nm, d in rc.items():
    if nm not in props: continue
    pr = props[nm]
    if pr['Q'] > 2 or pr['Inc'] < 30 or pr['Inc'] > 85: continue

    R = d['R']; Vo = d['Vo']; Vg = d['Vg']; Vd = d['Vd']; Vb = d['Vb']
    Vbsq = 0.5*Vd**2 + Vg*np.abs(Vg) + 0.7*Vb*np.abs(Vb)
    gb = np.where(R > 0, np.abs(Vbsq)*(1e3)**2 / (R*kpc_m), 1e-15)
    go = np.where(R > 0, (Vo*1e3)**2 / (R*kpc_m), 1e-15)
    ok = (gb > 1e-15) & (go > 1e-15) & (R > 0) & (Vo > 5)
    if ok.sum() < MIN_POINTS: continue

    si = np.argsort(R[ok]); Rs = R[ok][si]
    lgb = np.log10(gb[ok])[si]; lgo = np.log10(go[ok])[si]
    eps_raw = lgo - rar(lgb)
    n = len(eps_raw)
    Re = float(Rs[-1] - Rs[0])
    if Re <= 0: continue
    dR = float(np.median(np.diff(Rs)))
    med_lgb = float(np.median(lgb))

    galaxies.append({
        'nm': nm, 'n': n, 'Rs': Rs, 'eps_raw': eps_raw,
        'Re': Re, 'dR': dR, 'med_lgb': med_lgb,
        'Vflat': pr['Vflat'], 'Inc': pr['Inc'],
        'eD': pr['eD'], 'D': pr['D'],
    })

n_gal = len(galaxies)
print(f"  {n_gal} galaxies with N >= {MIN_POINTS}")


# ================================================================
# 3. FIT α FOR EACH DETRENDING MULTIPLIER
# ================================================================
print("\n[3] Fitting spectral slope α across detrending sweep...")

# Store results keyed by (mult, galaxy_index)
alpha_by_mult = {}

for mult in DETREND_MULTS:
    results_m = []
    n_fit = 0
    for gi, g in enumerate(galaxies):
        eps_det = detrend_residuals(g['Rs'], g['eps_raw'], mult)
        alpha, n_band, r2, resid_sc = fit_alpha(g['Rs'], eps_det, g['Re'], g['dR'])
        results_m.append({
            'alpha': alpha, 'n_band': n_band, 'r2': r2,
            'resid_scatter': resid_sc, 'eps_det': eps_det,
        })
        if alpha is not None:
            n_fit += 1
    alpha_by_mult[mult] = results_m
    print(f"  mult={mult}: {n_fit}/{n_gal} galaxies with usable α")


# ================================================================
# 4. PRIMARY (mult=0.5) AGGREGATE STATISTICS
# ================================================================
print("\n[4] Primary (s=n*var*0.5) aggregate statistics...")

prim = alpha_by_mult[PRIMARY_MULT]
alpha_vals = [prim[i]['alpha'] for i in range(n_gal) if prim[i]['alpha'] is not None]
alpha_arr = np.array(alpha_vals)
usable_idx = [i for i in range(n_gal) if prim[i]['alpha'] is not None]
n_usable = len(usable_idx)

print(f"  Usable galaxies: {n_usable}/{n_gal} ({n_usable/n_gal:.1%})")
print(f"  α: mean={np.mean(alpha_arr):.3f}, median={np.median(alpha_arr):.3f}, "
      f"std={np.std(alpha_arr):.3f}")
q25, q75 = np.percentile(alpha_arr, [25, 75])
print(f"  α IQR: [{q25:.3f}, {q75:.3f}]")
print(f"  α range: [{alpha_arr.min():.3f}, {alpha_arr.max():.3f}]")

# R² distribution
r2_vals = [prim[i]['r2'] for i in usable_idx]
print(f"  R² (fit quality): mean={np.mean(r2_vals):.3f}, median={np.median(r2_vals):.3f}")


# ================================================================
# 5. DETRENDING STABILITY
# ================================================================
print("\n[5] Detrending stability (cross-mult correlations)...")

print(f"\n  {'mult_a':>6} {'mult_b':>6} {'ρ':>7} {'p':>11} {'N_common':>9}")
print(f"  {'-'*48}")

stability_results = {}
for i, ma in enumerate(DETREND_MULTS):
    for mb in DETREND_MULTS[i+1:]:
        # Find galaxies with α in both
        common = [gi for gi in range(n_gal)
                  if alpha_by_mult[ma][gi]['alpha'] is not None
                  and alpha_by_mult[mb][gi]['alpha'] is not None]
        if len(common) < 10: continue
        a_a = np.array([alpha_by_mult[ma][gi]['alpha'] for gi in common])
        a_b = np.array([alpha_by_mult[mb][gi]['alpha'] for gi in common])
        rho, p = stats.spearmanr(a_a, a_b)
        print(f"  {ma:6.1f} {mb:6.1f} {rho:+7.3f} {p:11.4e} {len(common):9d}")
        stability_results[f'{ma}_vs_{mb}'] = {
            'rho': round(float(rho), 4), 'p': float(p), 'n': len(common)
        }


# ================================================================
# 6. CONFOUND CORRELATIONS
# ================================================================
print("\n[6] Confound correlations (α vs galaxy properties)...")

# Build arrays for usable galaxies
alpha_u = np.array([prim[i]['alpha'] for i in usable_idx])
npts_u  = np.array([galaxies[i]['n'] for i in usable_idx], dtype=float)
dR_u    = np.array([galaxies[i]['dR'] for i in usable_idx])
Re_u    = np.array([galaxies[i]['Re'] for i in usable_idx])
Vf_u    = np.array([galaxies[i]['Vflat'] for i in usable_idx])
Inc_u   = np.array([galaxies[i]['Inc'] for i in usable_idx])
eD_u    = np.array([galaxies[i]['eD'] for i in usable_idx])
mlgb_u  = np.array([galaxies[i]['med_lgb'] for i in usable_idx])
dRRe_u  = dR_u / Re_u

print(f"\n  {'Property':<14} {'ρ(α)':>7} {'p':>11} {'warn':>6}")
print(f"  {'-'*44}")

confound_results = {}
for pname, parr in [('N_pts', npts_u), ('med_dR', dR_u), ('R_extent', Re_u),
                     ('dR/R_ext', dRRe_u), ('Vflat', Vf_u), ('Inc', Inc_u),
                     ('eD', eD_u), ('med_log_gbar', mlgb_u)]:
    rho, p = stats.spearmanr(alpha_u, parr)
    warn = 'WARN' if abs(rho) > 0.3 and p < 0.05 else ''
    print(f"  {pname:<14} {rho:+7.3f} {p:11.4e} {warn:>6}")
    confound_results[pname] = {'rho': round(float(rho), 4), 'p': float(p)}


# ================================================================
# 7. SURROGATE NULL (parallelized)
# ================================================================
print(f"\n[7] Surrogate null ({N_SURR} per galaxy, {N_WORKERS} workers)...")
t0 = time.time()

# Prepare args for parallel processing
surr_args = []
for gi in usable_idx:
    g = galaxies[gi]
    eps_det = prim[gi]['eps_det']
    surr_args.append((
        g['Rs'], eps_det, g['Re'], g['dR'],
        N_SURR, 1000 + gi  # unique seed per galaxy
    ))

if N_WORKERS > 1:
    with Pool(N_WORKERS) as pool:
        surr_results_raw = pool.map(process_galaxy_surrogates, surr_args)
else:
    surr_results_raw = [process_galaxy_surrogates(a) for a in surr_args]

dt = time.time() - t0
print(f"  Surrogates done in {dt:.1f}s")

# Compute per-galaxy p-values
surr_pvals = []
surr_z_scores = []
n_sig = 0
per_galaxy_surr = []

for j, gi in enumerate(usable_idx):
    obs_alpha = prim[gi]['alpha']
    null_alphas = np.array(surr_results_raw[j])
    n_null = len(null_alphas)

    if n_null < 10:
        surr_pvals.append(np.nan)
        surr_z_scores.append(np.nan)
        per_galaxy_surr.append({'n_null': n_null, 'p': np.nan, 'z': np.nan})
        continue

    null_median = np.median(null_alphas)
    null_std = np.std(null_alphas)

    # Two-sided p-value: fraction of null with |α - median| >= |obs - median|
    obs_dev = abs(obs_alpha - null_median)
    null_devs = np.abs(null_alphas - null_median)
    p_val = float(np.mean(null_devs >= obs_dev))
    p_val = max(p_val, 1.0 / (n_null + 1))  # floor

    z = (obs_alpha - null_median) / null_std if null_std > 0 else 0.0

    surr_pvals.append(p_val)
    surr_z_scores.append(z)
    if p_val < 0.05:
        n_sig += 1

    per_galaxy_surr.append({
        'n_null': n_null,
        'null_median': round(float(null_median), 4),
        'null_std': round(float(null_std), 4),
        'p': round(float(p_val), 4),
        'z': round(float(z), 3),
    })

surr_pvals = np.array(surr_pvals)
surr_z_scores = np.array(surr_z_scores)
valid_surr = np.isfinite(surr_pvals)
n_valid_surr = int(valid_surr.sum())

frac_sig = n_sig / n_valid_surr if n_valid_surr > 0 else 0
expected_sig = 0.05 * n_valid_surr
binom_p = float(stats.binomtest(n_sig, n_valid_surr, 0.05, alternative='greater').pvalue) if n_valid_surr > 0 else 1.0

print(f"  Valid surrogates: {n_valid_surr}/{n_usable}")
print(f"  Galaxies with significant α (p<0.05): {n_sig}/{n_valid_surr} ({frac_sig:.1%})")
print(f"  Expected under null: {expected_sig:.1f} ({5:.0f}%)")
print(f"  Binomial p for excess: {binom_p:.4e}")

# Aggregate z-scores
valid_z = surr_z_scores[valid_surr]
print(f"  Mean z-score: {np.mean(valid_z):+.3f} ± {np.std(valid_z):.3f}")
print(f"  Median z-score: {np.median(valid_z):+.3f}")


# ================================================================
# 8. SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Galaxies analyzed: {n_gal}")
print(f"  Usable α (primary): {n_usable}/{n_gal} ({n_usable/n_gal:.1%})")
print(f"  α distribution: mean={np.mean(alpha_arr):.3f}, median={np.median(alpha_arr):.3f}, "
      f"std={np.std(alpha_arr):.3f}, IQR=[{q25:.3f}, {q75:.3f}]")

# Stability
print(f"\n  Detrending stability:")
for k, v in stability_results.items():
    print(f"    {k}: ρ = {v['rho']:+.3f}")

# Confounds
warns = [k for k, v in confound_results.items() if abs(v['rho']) > 0.3 and v['p'] < 0.05]
if warns:
    print(f"\n  CONFOUND WARNINGS: α correlates with {warns}")
else:
    print(f"\n  No significant confounds (all |ρ| < 0.3 or p > 0.05)")

# Surrogates
print(f"\n  Surrogate null:")
print(f"    Significant: {n_sig}/{n_valid_surr} ({frac_sig:.1%}), expected: {expected_sig:.1f}")
print(f"    Binomial p: {binom_p:.4e}")
print(f"    Mean z: {np.mean(valid_z):+.3f}")

# Verdict
if frac_sig > 0.20 and binom_p < 0.01:
    verdict = "SIGNIFICANT_SPECTRAL_SLOPE"
    print(f"\n  VERDICT: {verdict}")
    print(f"    Many galaxies have α significantly different from shuffled null.")
elif frac_sig > 0.10 and binom_p < 0.05:
    verdict = "MARGINAL_SPECTRAL_SLOPE"
    print(f"\n  VERDICT: {verdict}")
    print(f"    Modest excess of significant α values.")
else:
    verdict = "NO_SPECTRAL_SLOPE_EXCESS"
    print(f"\n  VERDICT: {verdict}")
    print(f"    α distribution consistent with shuffled null.")


# ================================================================
# 9. SAVE JSON
# ================================================================
per_galaxy_table = []
for j, gi in enumerate(usable_idx):
    g = galaxies[gi]
    p = prim[gi]
    s = per_galaxy_surr[j]
    per_galaxy_table.append({
        'name': g['nm'],
        'n_pts': g['n'],
        'R_extent': round(g['Re'], 3),
        'med_dR': round(g['dR'], 4),
        'dR_Re': round(g['dR'] / g['Re'], 4),
        'Vflat': g['Vflat'],
        'Inc': g['Inc'],
        'eD': g['eD'],
        'med_log_gbar': round(g['med_lgb'], 3),
        'alpha': round(p['alpha'], 4),
        'n_band': p['n_band'],
        'r2': round(p['r2'], 4) if p['r2'] is not None else None,
        'resid_scatter': round(p['resid_scatter'], 4) if p['resid_scatter'] is not None else None,
        'surr_p': s['p'],
        'surr_z': s['z'],
        'surr_null_median': s.get('null_median'),
        'surr_null_std': s.get('null_std'),
    })

# α for other detrending mults
alpha_sweep = {}
for mult in DETREND_MULTS:
    vals = [alpha_by_mult[mult][i]['alpha'] for i in usable_idx
            if alpha_by_mult[mult][i]['alpha'] is not None]
    alpha_sweep[str(mult)] = {
        'n_usable': len(vals),
        'mean': round(float(np.mean(vals)), 4) if vals else None,
        'median': round(float(np.median(vals)), 4) if vals else None,
        'std': round(float(np.std(vals)), 4) if vals else None,
    }

results = {
    'test': 'spectral_slope_alpha',
    'description': 'Per-galaxy power-spectrum slope α from LS periodogram of detrended RAR residuals.',
    'parameters': {
        'min_points': MIN_POINTS,
        'n_freq': N_FREQ,
        'min_band_pts': MIN_BAND_PTS,
        'n_surrogates': N_SURR,
        'detrend_mults': DETREND_MULTS,
        'primary_mult': PRIMARY_MULT,
        'fit_method': 'Theil-Sen',
    },
    'sample': {
        'n_galaxies': n_gal,
        'n_usable': n_usable,
        'frac_usable': round(n_usable / n_gal, 3),
    },
    'alpha_distribution': {
        'mean': round(float(np.mean(alpha_arr)), 4),
        'median': round(float(np.median(alpha_arr)), 4),
        'std': round(float(np.std(alpha_arr)), 4),
        'q25': round(float(q25), 4),
        'q75': round(float(q75), 4),
        'min': round(float(alpha_arr.min()), 4),
        'max': round(float(alpha_arr.max()), 4),
    },
    'fit_quality': {
        'r2_mean': round(float(np.mean(r2_vals)), 4),
        'r2_median': round(float(np.median(r2_vals)), 4),
    },
    'detrending_sweep': alpha_sweep,
    'detrending_stability': stability_results,
    'confound_correlations': confound_results,
    'surrogate_null': {
        'n_valid': n_valid_surr,
        'n_significant_005': n_sig,
        'frac_significant': round(frac_sig, 4),
        'expected_null': round(expected_sig, 1),
        'binom_p_excess': binom_p,
        'mean_z': round(float(np.mean(valid_z)), 4),
        'median_z': round(float(np.median(valid_z)), 4),
        'std_z': round(float(np.std(valid_z)), 4),
    },
    'verdict': verdict,
    'per_galaxy': per_galaxy_table,
}

out_json = os.path.join(RESULTS_DIR, 'summary_spectral_slope_alpha.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved JSON: {out_json}")


# ================================================================
# 10. SAVE CSV
# ================================================================
out_csv = os.path.join(RESULTS_DIR, 'spectral_slope_alpha_per_galaxy.csv')
if per_galaxy_table:
    fieldnames = list(per_galaxy_table[0].keys())
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_galaxy_table)
    print(f"Saved CSV: {out_csv}")

print("=" * 72)
