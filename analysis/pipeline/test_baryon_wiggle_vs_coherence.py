#!/usr/bin/env python3
"""
Baryon Wiggle Index vs Residual Coherence (Step 29)
====================================================

Test whether RAR residual autocorrelation is driven by baryonic structure
(bumps in gbar(r)) or by something else ("interface physics").

A) Compute per-galaxy "wiggle index" from gbar(r):
   - W1: RMS of 2nd derivative of spline-smoothed log gbar(r)
   - W2: RMS of first-differences of log gbar(r) (simpler, no smoothing)

B) Correlate wiggle index with:
   - r1_raw   (lag-1 ACF of raw RAR residuals)
   - r1_det   (lag-1 ACF of detrended residuals)
   - LS peak power / p-value (Lomb-Scargle on detrended residuals)

C) Partial correlations controlling N_pts and dR/R_ext.

INTERPRETATION:
  - Strong W vs r1 → baryon-driven structure (still interesting, different claim)
  - Weak W vs r1  → big win for "interface physics"

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os, json, numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

g_dagger_SI  = 1.20e-10
kpc_m        = 3.086e19
MIN_POINTS   = 15
DETREND_MULT = 0.5
N_SURR_LS    = 200   # LS surrogates per galaxy (matches Step 27)

np.random.seed(42)

print("=" * 72)
print("BARYON WIGGLE INDEX vs RESIDUAL COHERENCE (Step 29)")
print("=" * 72)

# ---------- helpers ----------
def rar(lg):
    gb = 10.**lg
    return np.log10(gb / (1. - np.exp(-np.sqrt(gb / 1.2e-10))))

def lag_acf(x, k=1):
    n = len(x)
    if n <= k + 1: return np.nan
    m = np.mean(x); v = np.var(x, ddof=0)
    if v < 1e-30: return np.nan
    return np.mean((x[:n-k] - m) * (x[k:] - m)) / v

def partial_spearman(x, y, *covariates):
    from scipy.stats import rankdata, spearmanr
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    Z = np.column_stack([rankdata(c).astype(float) for c in covariates])
    Z1 = np.column_stack([np.ones(len(rx)), Z])
    bx, _, _, _ = np.linalg.lstsq(Z1, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(Z1, ry, rcond=None)
    rho, p = spearmanr(rx - Z1 @ bx, ry - Z1 @ by)
    return round(float(rho), 4), float(p)


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
            'D': float(p[1]), 'Inc': float(p[4]),
            'L36': float(p[6]), 'Vflat': float(p[14]),
            'Q': int(p[16]),
        }
    except: continue


# ================================================================
# 2. COMPUTE PER-GALAXY METRICS
# ================================================================
print("\n[2] Computing wiggle index + coherence metrics per galaxy...")

perm_rng = np.random.default_rng(789)
gals = []

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
    n = len(Rs)
    Re = float(Rs[-1] - Rs[0])
    if Re <= 0: continue
    dR = float(np.median(np.diff(Rs)))

    # --- RAR residuals ---
    eps_raw = lgo - rar(lgb)

    # --- Detrend ---
    v = np.var(eps_raw); s = n * v * DETREND_MULT
    try:
        sp = UnivariateSpline(Rs, eps_raw, k=min(3, n-1), s=s)
        eps_det = eps_raw - sp(Rs)
    except:
        eps_det = eps_raw - np.mean(eps_raw)

    # --- Coherence metrics ---
    r1_raw = lag_acf(eps_raw, 1)
    r1_det = lag_acf(eps_det, 1)

    # --- Wiggle index W1: RMS of 2nd derivative of smoothed log gbar ---
    # Use a mild spline on log gbar(R) to capture real structure
    try:
        var_lgb = np.var(lgb)
        s_gbar = n * var_lgb * 1.0   # mild smoothing
        sp_gb = UnivariateSpline(Rs, lgb, k=min(3, n-1), s=s_gbar)
        # 2nd derivative at data points
        d2 = sp_gb.derivative(n=2)(Rs)
        W1 = float(np.sqrt(np.mean(d2**2)))
    except:
        W1 = np.nan

    # --- Wiggle index W2: RMS of first differences of log gbar ---
    dlgb = np.diff(lgb) / np.diff(Rs)  # d(log gbar)/dR
    W2 = float(np.sqrt(np.mean(dlgb**2)))

    # --- Wiggle index W3: RMS of 2nd finite-differences (discrete curvature) ---
    if n >= 3:
        d2_fd = np.diff(dlgb) / (0.5 * (np.diff(Rs)[:-1] + np.diff(Rs)[1:]))
        W3 = float(np.sqrt(np.mean(d2_fd**2)))
    else:
        W3 = np.nan

    # --- Lomb-Scargle on detrended residuals ---
    std_eps = np.std(eps_det)
    if std_eps > 1e-30:
        y_ls = (eps_det - np.mean(eps_det)) / std_eps
        f_min = 1.0 / Re; f_max = (n / 2.0) / Re
        n_freq = min(500, 10 * n)
        freq_grid = np.linspace(f_min, f_max, n_freq)
        ls = LombScargle(Rs, y_ls, fit_mean=False, center_data=True)
        power = ls.power(freq_grid)
        idx_pk = np.argmax(power)
        ls_power = float(power[idx_pk])
        ls_freq  = float(freq_grid[idx_pk])
        ls_wl    = 1.0 / ls_freq
        # Permutation null for p-value
        null_peaks = np.zeros(N_SURR_LS)
        for ss in range(N_SURR_LS):
            y_shuf = perm_rng.permutation(y_ls)
            ls_null = LombScargle(Rs, y_shuf, fit_mean=False, center_data=True)
            null_peaks[ss] = np.max(ls_null.power(freq_grid))
        ls_p = float(np.mean(null_peaks >= ls_power))
    else:
        ls_power = ls_p = ls_wl = np.nan

    gals.append({
        'nm': nm, 'n': n, 'Re': Re, 'dR': dR,
        'r1_raw': r1_raw, 'r1_det': r1_det,
        'W1': W1, 'W2': W2, 'W3': W3,
        'ls_power': ls_power, 'ls_p': ls_p, 'ls_wl': ls_wl,
    })

nv = len(gals)
print(f"  {nv} galaxies computed")

# Extract arrays
r1_raw_arr = np.array([g['r1_raw'] for g in gals])
r1_det_arr = np.array([g['r1_det'] for g in gals])
W1_arr     = np.array([g['W1']     for g in gals])
W2_arr     = np.array([g['W2']     for g in gals])
W3_arr     = np.array([g['W3']     for g in gals])
ls_pow_arr = np.array([g['ls_power'] for g in gals])
ls_p_arr   = np.array([g['ls_p']   for g in gals])
N_arr      = np.array([g['n']      for g in gals], dtype=float)
Re_arr     = np.array([g['Re']     for g in gals])
dR_arr     = np.array([g['dR']     for g in gals])
dR_Re_arr  = dR_arr / Re_arr
log_N      = np.log10(N_arr)
log_dRRe   = np.log10(dR_Re_arr)
log_W1     = np.log10(np.where(W1_arr > 0, W1_arr, np.nan))
log_W2     = np.log10(np.where(W2_arr > 0, W2_arr, np.nan))
log_W3     = np.log10(np.where(W3_arr > 0, W3_arr, np.nan))

# LS significance flag
ls_sig_arr = (ls_p_arr < 0.05).astype(float)
n_ls_sig = int(np.nansum(ls_sig_arr))
print(f"  LS significant peaks: {n_ls_sig}/{nv} ({n_ls_sig/nv:.1%})")


# ================================================================
# 3. BIVARIATE CORRELATIONS: WIGGLE vs COHERENCE
# ================================================================
print("\n" + "=" * 72)
print("TEST A: BIVARIATE CORRELATIONS (Spearman)")
print("=" * 72)

print(f"\n  {'Wiggle':<8} {'Coherence':<12} {'ρ':>7} {'p':>11} {'N':>4}")
print(f"  {'-'*50}")

biv_results = {}
for wname, warr in [('W1', W1_arr), ('W2', W2_arr), ('W3', W3_arr),
                     ('logW1', log_W1), ('logW2', log_W2), ('logW3', log_W3)]:
    for cname, carr in [('r1_raw', r1_raw_arr), ('r1_det', r1_det_arr),
                        ('ls_power', ls_pow_arr)]:
        ok = np.isfinite(warr) & np.isfinite(carr)
        if ok.sum() < 10: continue
        rho, p = stats.spearmanr(warr[ok], carr[ok])
        sig = '*' if p < 0.05 else ' '
        print(f"  {wname:<8} {cname:<12} {rho:+7.3f} {p:11.4e}{sig} {ok.sum():4d}")
        biv_results[f'{wname}_vs_{cname}'] = {
            'rho': round(float(rho), 4), 'p': float(p), 'n': int(ok.sum())
        }


# ================================================================
# 4. PARTIAL CORRELATIONS: CONTROLLING N_pts AND dR/R_ext
# ================================================================
print("\n" + "=" * 72)
print("TEST B: PARTIAL CORRELATIONS | N_pts, dR/R_ext")
print("=" * 72)

print(f"\n  {'Wiggle':<8} {'Coherence':<12} {'ρ_part':>7} {'p':>11} {'N':>4}")
print(f"  {'-'*50}")

part_results = {}
for wname, warr in [('logW1', log_W1), ('logW2', log_W2), ('logW3', log_W3)]:
    for cname, carr in [('r1_raw', r1_raw_arr), ('r1_det', r1_det_arr),
                        ('ls_power', ls_pow_arr)]:
        ok = np.isfinite(warr) & np.isfinite(carr)
        if ok.sum() < 15: continue
        rho, p = partial_spearman(warr[ok], carr[ok], log_N[ok], log_dRRe[ok])
        sig = '*' if p < 0.05 else ' '
        print(f"  {wname:<8} {cname:<12} {rho:+7.3f} {p:11.4e}{sig} {ok.sum():4d}")
        part_results[f'{wname}_vs_{cname}'] = {
            'rho': round(float(rho), 4), 'p': float(p), 'n': int(ok.sum())
        }


# ================================================================
# 5. WIGGLE vs ξ/R (CONTROL: IS WIGGLE JUST MASS?)
# ================================================================
print("\n" + "=" * 72)
print("TEST C: WIGGLE vs ξ/R_ext (mass confound check)")
print("=" * 72)

G_kpc = 4.30091e-6
g_dagger_kpc = g_dagger_SI * kpc_m / 1e6
Msun_L36 = 0.5; HELIUM = 1.33

# Reload Mb for ξ computation
xi_Re_arr = np.full(nv, np.nan)
for i, g in enumerate(gals):
    nm = g['nm']
    if nm in props:
        pr = props[nm]
        L36 = pr['L36']
        # Need MHI — reload from MRT
        # Actually we need it from the MRT which has MHI at index 12
        pass

# Simpler: reparse MHI from MRT for galaxies in our sample
mrt_mhi = {}
for ln in lines[ds:]:
    if not ln.strip(): continue
    try:
        nm = ln[0:11].strip(); p = ln[11:].split()
        if len(p) < 17: continue
        mrt_mhi[nm] = float(p[12])
    except: continue

for i, g in enumerate(gals):
    nm = g['nm']
    if nm in props and nm in mrt_mhi:
        pr = props[nm]
        Mb = Msun_L36 * pr['L36'] * 1e9 + HELIUM * mrt_mhi[nm] * 1e9
        if Mb > 0:
            xi = np.sqrt(G_kpc * Mb / g_dagger_kpc)
            xi_Re_arr[i] = xi / g['Re']

log_xiRe = np.log10(np.where(xi_Re_arr > 0, xi_Re_arr, np.nan))
ok_xi = np.isfinite(log_xiRe)

print(f"\n  Galaxies with ξ/R: {ok_xi.sum()}/{nv}")

# Wiggle vs ξ/R
print(f"\n  {'Metric':<8} {'vs':>12} {'ρ':>7} {'p':>11}")
print(f"  {'-'*42}")
xi_corrs = {}
for wname, warr in [('logW1', log_W1), ('logW2', log_W2), ('logW3', log_W3)]:
    ok = np.isfinite(warr) & ok_xi
    if ok.sum() < 10: continue
    rho, p = stats.spearmanr(warr[ok], log_xiRe[ok])
    sig = '*' if p < 0.05 else ' '
    print(f"  {wname:<8} {'ξ/R_ext':>12} {rho:+7.3f} {p:11.4e}{sig}")
    xi_corrs[f'{wname}_vs_xiRe'] = {'rho': round(float(rho), 4), 'p': float(p)}


# ================================================================
# 6. KEY TEST: DOES r1 SURVIVE AFTER CONTROLLING WIGGLE?
# ================================================================
print("\n" + "=" * 72)
print("TEST D: r1 vs ξ/R_ext CONTROLLING WIGGLE + N + dR/R")
print("=" * 72)

print(f"\n  If ξ/R signal survives after controlling wiggle → interface physics")
print(f"  If ξ/R signal dies after controlling wiggle → baryon structure\n")

print(f"  {'Coherence':<10} {'Wiggle ctrl':<10} {'ρ':>7} {'p':>11} {'N':>4}")
print(f"  {'-'*50}")

key_results = {}
for cname, carr in [('r1_raw', r1_raw_arr), ('r1_det', r1_det_arr)]:
    for wname, warr in [('logW1', log_W1), ('logW2', log_W2), ('logW3', log_W3)]:
        ok = np.isfinite(warr) & np.isfinite(carr) & ok_xi
        if ok.sum() < 15: continue

        # Partial: coherence vs ξ/R | N, dR/R, Wiggle
        rho, p = partial_spearman(carr[ok], log_xiRe[ok],
                                   log_N[ok], log_dRRe[ok], warr[ok])
        sig = '*' if p < 0.05 else ' '
        print(f"  {cname:<10} {wname:<10} {rho:+7.3f} {p:11.4e}{sig} {ok.sum():4d}")
        key_results[f'{cname}_vs_xiRe_ctrl_{wname}'] = {
            'rho': round(float(rho), 4), 'p': float(p), 'n': int(ok.sum())
        }

    # Baseline: without wiggle control
    ok_base = np.isfinite(carr) & ok_xi
    if ok_base.sum() >= 15:
        rho_base, p_base = partial_spearman(carr[ok_base], log_xiRe[ok_base],
                                             log_N[ok_base], log_dRRe[ok_base])
        sig = '*' if p_base < 0.05 else ' '
        print(f"  {cname:<10} {'(none)':10} {rho_base:+7.3f} {p_base:11.4e}{sig} {ok_base.sum():4d}  ← baseline")
        key_results[f'{cname}_vs_xiRe_baseline'] = {
            'rho': round(float(rho_base), 4), 'p': float(p_base), 'n': int(ok_base.sum())
        }


# ================================================================
# 7. SUMMARY + INTERPRETATION
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

# Find strongest wiggle-coherence correlation
best_biv = max(biv_results.items(), key=lambda x: abs(x[1]['rho']))
best_part = max(part_results.items(), key=lambda x: abs(x[1]['rho'])) if part_results else (None, None)

print(f"\n  Strongest bivariate (wiggle vs coherence):")
print(f"    {best_biv[0]}: ρ = {best_biv[1]['rho']:+.3f}, p = {best_biv[1]['p']:.4e}")

if best_part[0]:
    print(f"  Strongest partial (| N, dR/R):")
    print(f"    {best_part[0]}: ρ = {best_part[1]['rho']:+.3f}, p = {best_part[1]['p']:.4e}")

# Key diagnostic: does wiggle correlate strongly with r1?
# Use logW2 vs r1_det as primary (simplest, most robust)
ok_prim = np.isfinite(log_W2) & np.isfinite(r1_det_arr)
if ok_prim.sum() >= 10:
    rho_wig, p_wig = stats.spearmanr(log_W2[ok_prim], r1_det_arr[ok_prim])
else:
    rho_wig, p_wig = np.nan, np.nan

print(f"\n  PRIMARY DIAGNOSTIC (logW2 vs r1_det):")
print(f"    ρ = {rho_wig:+.3f}, p = {p_wig:.4e}")

if abs(rho_wig) > 0.4 and p_wig < 0.01:
    verdict = "BARYON_DRIVEN"
    print(f"\n  VERDICT: {verdict}")
    print(f"    Wiggle index tracks r1 hard → coherence is baryon-structure driven")
    print(f"    (Still interesting for structure formation, but different BEC claim)")
elif abs(rho_wig) < 0.2 or p_wig > 0.10:
    verdict = "INTERFACE_PHYSICS"
    print(f"\n  VERDICT: {verdict}")
    print(f"    Wiggle index does NOT predict r1 → coherence is NOT baryon-driven")
    print(f"    Big win for interface/condensate physics interpretation")
else:
    verdict = "MIXED"
    print(f"\n  VERDICT: {verdict}")
    print(f"    Moderate wiggle-coherence coupling — partially baryon-driven")
    print(f"    Need further tests to separate contributions")

# Check if ξ/R survives wiggle control
baseline_keys = [k for k in key_results if 'baseline' in k and 'r1_det' in k]
wiggle_keys = [k for k in key_results if 'ctrl_logW2' in k and 'r1_det' in k]
if baseline_keys and wiggle_keys:
    bl = key_results[baseline_keys[0]]
    wc = key_results[wiggle_keys[0]]
    print(f"\n  ξ/R signal check:")
    print(f"    Baseline (no wiggle ctrl):  ρ = {bl['rho']:+.3f}, p = {bl['p']:.4e}")
    print(f"    With wiggle control (W2):   ρ = {wc['rho']:+.3f}, p = {wc['p']:.4e}")
    if wc['p'] < 0.05:
        print(f"    → ξ/R SURVIVES wiggle control ✓")
    elif wc['p'] < 0.10:
        print(f"    → ξ/R marginal after wiggle control")
    else:
        print(f"    → ξ/R killed by wiggle control")


# ================================================================
# SAVE
# ================================================================
results = {
    'test': 'baryon_wiggle_vs_coherence',
    'description': 'Test whether RAR residual autocorrelation is driven by baryonic structure.',
    'sample': {'n_galaxies': nv, 'n_ls_significant': n_ls_sig},
    'wiggle_stats': {
        'W1_median': round(float(np.nanmedian(W1_arr)), 6),
        'W2_median': round(float(np.nanmedian(W2_arr)), 6),
        'W3_median': round(float(np.nanmedian(W3_arr)), 6),
    },
    'bivariate_correlations': biv_results,
    'partial_correlations_NdRR': part_results,
    'wiggle_vs_xiRe': xi_corrs,
    'xiRe_signal_with_wiggle_control': key_results,
    'primary_diagnostic': {
        'metric': 'logW2_vs_r1_det',
        'rho': round(float(rho_wig), 4),
        'p': float(p_wig),
    },
    'verdict': verdict,
}

out = os.path.join(RESULTS_DIR, 'summary_baryon_wiggle_vs_coherence.json')
with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {out}")
print("=" * 72)
