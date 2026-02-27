#!/usr/bin/env python3
"""
Healing-Length Scaling — continuous ACF (Step 28e)
===================================================

Metrics: r1, AR1, AR1_grid (resampled to 0.5 kpc common grid).
Controls: dR/R_ext, N_pts, Inc, eD (distance uncertainty).
Detrend sweep.  BH-FDR across partials.
Verdict keyed on ξ/R_ext partial.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os, json, numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

g_dagger_SI  = 1.20e-10
kpc_m        = 3.086e19
G_kpc        = 4.30091e-6
g_dagger_kpc = g_dagger_SI * kpc_m / 1e6
Msun_L36     = 0.5
HELIUM       = 1.33
MIN_POINTS   = 15
N_PERM       = 5000
N_BOOT       = 2000
DETREND_MULTS = [0.25, 0.5, 1.0, 2.0]
GRID_STEP     = 0.5   # kpc, for common-grid resampling
MIN_GRID_PTS  = 8     # minimum grid points for AR1_grid

np.random.seed(42)

print("=" * 72)
print("HEALING-LENGTH ACF TEST (Step 28e)")
print("  + AR1_grid (0.5 kpc) | Inc/eD controls | BH-FDR")
print("=" * 72)

# ---------- env ----------
UMA = {'NGC3726','NGC3769','NGC3877','NGC3893','NGC3917','NGC3949','NGC3953',
       'NGC3972','NGC3992','NGC4010','NGC4013','NGC4051','NGC4085','NGC4088',
       'NGC4100','NGC4138','NGC4157','NGC4183','NGC4217','UGC06399','UGC06446',
       'UGC06667','UGC06786','UGC06787','UGC06818','UGC06917','UGC06923',
       'UGC06930','UGC06973','UGC06983','UGC07089'}
GRP = {'NGC2403','NGC2976','IC2574','DDO154','DDO168','UGC04483',
       'NGC0300','NGC0055','NGC0247','NGC7793','NGC2915','UGCA442',
       'ESO444-G084','UGC07577','UGC07232','NGC3741','NGC4068',
       'UGC07866','UGC07524','UGC08490','UGC07559','NGC3109','NGC5055'}
def classify_env(n): return 'dense' if n in UMA or n in GRP else 'field'

def rar(lg):
    gb = 10.**lg; return np.log10(gb / (1. - np.exp(-np.sqrt(gb / 1.2e-10))))

def lag_acf(x, k=1):
    n = len(x)
    if n <= k + 1: return np.nan
    m = np.mean(x); v = np.var(x, ddof=0)
    if v < 1e-30: return np.nan
    return np.mean((x[:n-k] - m) * (x[k:] - m)) / v

def ar1_coeff(x):
    n = len(x)
    if n < 5: return np.nan
    y = x[1:]; X = np.column_stack([np.ones(n-1), x[:-1]])
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return float(b[1])
    except: return np.nan

def resample_to_grid(R, eps, step=0.5, min_pts=8):
    """Resample irregularly-spaced residuals onto a uniform radial grid.
    Uses linear interpolation, then computes AR(1)."""
    R_min, R_max = R[0], R[-1]
    n_grid = int(np.floor((R_max - R_min) / step)) + 1
    if n_grid < min_pts: return np.nan
    R_grid = np.linspace(R_min, R_min + (n_grid - 1) * step, n_grid)
    eps_grid = np.interp(R_grid, R, eps)
    return ar1_coeff(eps_grid)

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

def ols_regression(X, y, names):
    Xa = np.column_stack([np.ones(len(y)), X])
    b, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
    yp = Xa @ b; ssr = np.sum((y - yp)**2); sst = np.sum((y - np.mean(y))**2)
    r2 = 1 - ssr/sst if sst > 0 else 0
    n, k = Xa.shape; mse = ssr / max(n - k, 1)
    try:
        cov = mse * np.linalg.inv(Xa.T @ Xa); se = np.sqrt(np.diag(cov))
    except: se = np.full(k, np.nan)
    t = b / np.where(se > 0, se, 1)
    pv = 2 * stats.t.sf(np.abs(t), df=max(n - k, 1))
    allnames = ['intercept'] + names
    return {cn: {'beta': round(float(b[i]),4), 'se': round(float(se[i]),4),
                 't': round(float(t[i]),2), 'p': round(float(pv[i]),4)}
            for i, cn in enumerate(allnames)}, round(float(r2), 4)

def benjamini_hochberg(pvals, alpha=0.05):
    """Returns array of booleans (reject?) and adjusted p-values."""
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[idx]
    adjusted = np.zeros(n)
    for i in range(n-1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i+1], sorted_p[i] * n / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)
    # Map back to original order
    adj_orig = np.zeros(n)
    adj_orig[idx] = adjusted
    return adj_orig < alpha, adj_orig


# ================================================================
# 1. LOAD SPARC DATA (with Inc, eD, Q)
# ================================================================
print("\n[1] Loading SPARC...")
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
        if nm not in rc: rc[nm] = {'R':[],'Vo':[],'Vg':[],'Vd':[],'Vb':[]}
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
        L36 = float(p[6]); MHI = float(p[12]); Rd = float(p[10])
        Mb = Msun_L36 * L36 * 1e9 + HELIUM * MHI * 1e9
        props[nm] = {
            'Inc': float(p[4]), 'eD': float(p[2]), 'Q': int(p[16]),
            'Vf': float(p[14]), 'Mb': Mb, 'Rd': Rd,
        }
    except: continue

print(f"  {len(rc)} RCs, {len(props)} props")


# ================================================================
# 2. BUILD RAW RESIDUALS
# ================================================================
print("\n[2] Computing raw RAR residuals...")
gals = []
for nm, d in rc.items():
    if nm not in props: continue
    pr = props[nm]
    if pr['Q'] > 2 or pr['Inc'] < 30 or pr['Inc'] > 85 or pr['Mb'] <= 0: continue
    R = d['R']; Vo = d['Vo']; Vg = d['Vg']; Vd = d['Vd']; Vb = d['Vb']
    Vbsq = 0.5*Vd**2 + Vg*np.abs(Vg) + 0.7*Vb*np.abs(Vb)
    gb = np.where(R > 0, np.abs(Vbsq)*(1e3)**2 / (R*kpc_m), 1e-15)
    go = np.where(R > 0, (Vo*1e3)**2 / (R*kpc_m), 1e-15)
    ok = (gb > 1e-15) & (go > 1e-15) & (R > 0) & (Vo > 5)
    if ok.sum() < MIN_POINTS: continue
    si = np.argsort(R[ok]); Rs = R[ok][si]
    lgb = np.log10(gb[ok])[si]; lgo = np.log10(go[ok])[si]
    eps = lgo - rar(lgb); n = len(eps)
    Re = float(Rs[-1] - Rs[0])
    if Re <= 0: continue
    dR = float(np.median(np.diff(Rs)))
    xi = np.sqrt(G_kpc * pr['Mb'] / g_dagger_kpc)
    gals.append({'nm':nm, 'n':n, 'R':Rs, 'eps':eps,
                 'Re':Re, 'dR':dR, 'Vf':pr['Vf'],
                 'Mb':pr['Mb'], 'xi':xi, 'env':classify_env(nm),
                 'Inc':pr['Inc'], 'eD':pr['eD'], 'Q':pr['Q']})

print(f"  {len(gals)} galaxies")


# ================================================================
# 3. DETREND + COMPUTE METRICS
# ================================================================
print(f"\n[3] Detrending sweep: s = n*var * {DETREND_MULTS}")
print(f"    + AR1_grid (step={GRID_STEP} kpc)")

def do_detrend(R, eps, mult):
    n = len(eps); v = np.var(eps); s = n * v * mult
    try:
        sp = UnivariateSpline(R, eps, k=min(3, n-1), s=s)
        return eps - sp(R)
    except: return eps - np.mean(eps)

for mult in DETREND_MULTS:
    for g in gals:
        ed = do_detrend(g['R'], g['eps'], mult)
        g[f'r1_{mult}'] = lag_acf(ed, 1)
        g[f'r2_{mult}'] = lag_acf(ed, 2)
        g[f'ar1_{mult}'] = ar1_coeff(ed)
        g[f'ar1g_{mult}'] = resample_to_grid(g['R'], ed, step=GRID_STEP,
                                              min_pts=MIN_GRID_PTS)

PRIMARY = 0.5


# ================================================================
# 4. EXTRACT ARRAYS
# ================================================================
nv = len(gals)
r1    = np.array([g[f'r1_{PRIMARY}']   for g in gals])
r2    = np.array([g[f'r2_{PRIMARY}']   for g in gals])
ar1   = np.array([g[f'ar1_{PRIMARY}']  for g in gals])
ar1g  = np.array([g[f'ar1g_{PRIMARY}'] for g in gals])
xi    = np.array([g['xi']  for g in gals])
Re    = np.array([g['Re']  for g in gals])
dR    = np.array([g['dR']  for g in gals])
Npts  = np.array([g['n']   for g in gals], dtype=float)
Vf    = np.array([g['Vf']  for g in gals])
Inc   = np.array([g['Inc'] for g in gals])
eD    = np.array([g['eD']  for g in gals])
xi_Re = xi / Re
dR_Re = dR / Re

log_xi   = np.log10(xi)
log_Re   = np.log10(Re)
log_xiRe = np.log10(xi_Re)
log_N    = np.log10(Npts)
log_dRRe = np.log10(dR_Re)

ar1g_valid = ~np.isnan(ar1g)
n_grid_valid = int(ar1g_valid.sum())

print(f"\n[4] {nv} galaxies, primary detrend mult={PRIMARY}")
print(f"  r1:      mean={np.mean(r1):.3f}  median={np.median(r1):.3f}")
print(f"  AR1:     mean={np.nanmean(ar1):.3f}  median={np.nanmedian(ar1):.3f}")
print(f"  AR1_grid: {n_grid_valid} valid, mean={np.nanmean(ar1g):.3f}  median={np.nanmedian(ar1g):.3f}")
print(f"  Inc: mean={np.mean(Inc):.1f}°  eD: mean={np.mean(eD):.2f} Mpc")


# ================================================================
# 5. BIVARIATE CORRELATIONS
# ================================================================
print("\n[5] Spearman correlations (primary detrend)...")

def sp(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10: return np.nan, np.nan
    return stats.spearmanr(a[m], b[m])

corrs = {}
metrics_list = [('r1', r1), ('AR1', ar1), ('AR1_grid', ar1g)]
preds_list   = [('xi_Re', xi_Re), ('xi', xi), ('Rext', Re),
                ('Npts', Npts), ('dR_Re', dR_Re), ('Inc', Inc), ('eD', eD)]

print(f"\n  {'metric':<10} {'vs':<8} {'ρ':>8} {'p':>12}")
print(f"  {'-'*40}")
for mname, marr in metrics_list:
    for pname, parr in preds_list:
        rho, p = sp(marr, parr)
        if np.isnan(rho): continue
        rho, p = round(float(rho), 4), float(p)
        corrs[f'{mname}_vs_{pname}'] = {'rho': rho, 'p': p}
        sig = '*' if p < 0.05 else ' '
        print(f"  {mname:<10} {pname:<8} {rho:+8.3f} {p:12.4e} {sig}")


# ================================================================
# 6. PARTIAL CORRELATIONS — full battery
# ================================================================
print("\n[6] Partial Spearman correlations...")

partial_tests = {}

# -- Core partials (r1) --
rho, p = partial_spearman(r1, log_xiRe, log_N, log_dRRe)
partial_tests['r1_vs_xiRe | N,dRRe'] = {'rho': rho, 'p': p}

rho, p = partial_spearman(r1, log_xiRe, log_N, log_dRRe, Inc)
partial_tests['r1_vs_xiRe | N,dRRe,Inc'] = {'rho': rho, 'p': p}

rho, p = partial_spearman(r1, log_xiRe, log_N, log_dRRe, Inc, eD)
partial_tests['r1_vs_xiRe | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

# -- Negatives (should be null) --
rho, p = partial_spearman(r1, log_xi, log_N, log_dRRe, Inc, eD)
partial_tests['r1_vs_xi   | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

rho, p = partial_spearman(r1, log_Re, log_N, log_dRRe, Inc, eD)
partial_tests['r1_vs_Rext | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

# -- AR1 partials --
rho, p = partial_spearman(ar1, log_xiRe, log_N, log_dRRe, Inc, eD)
partial_tests['AR1_vs_xiRe | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

# -- AR1_grid partials (N-independent metric) --
v = ar1g_valid
if v.sum() >= 20:
    rho, p = partial_spearman(ar1g[v], log_xiRe[v], log_N[v], log_dRRe[v])
    partial_tests['AR1g_vs_xiRe | N,dRRe'] = {'rho': rho, 'p': p}

    rho, p = partial_spearman(ar1g[v], log_xiRe[v], log_N[v], log_dRRe[v], Inc[v], eD[v])
    partial_tests['AR1g_vs_xiRe | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

    rho, p = partial_spearman(ar1g[v], log_xi[v], log_N[v], log_dRRe[v], Inc[v], eD[v])
    partial_tests['AR1g_vs_xi   | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

    rho, p = partial_spearman(ar1g[v], log_Re[v], log_N[v], log_dRRe[v], Inc[v], eD[v])
    partial_tests['AR1g_vs_Rext | N,dRRe,Inc,eD'] = {'rho': rho, 'p': p}

    # AR1_grid with NO confound controls (bivariate for comparison)
    rho_biv, p_biv = stats.spearmanr(ar1g[v], xi_Re[v])
    partial_tests['AR1g_vs_xiRe | (none)'] = {'rho': round(float(rho_biv),4), 'p': float(p_biv)}

# -- BH-FDR correction --
test_names = list(partial_tests.keys())
raw_pvals = [partial_tests[k]['p'] for k in test_names]
reject, adj_pvals = benjamini_hochberg(raw_pvals)

for i, k in enumerate(test_names):
    partial_tests[k]['p_adj_bh'] = round(float(adj_pvals[i]), 4)
    partial_tests[k]['reject_bh'] = bool(reject[i])

print(f"\n  {'Test':<38} {'ρ':>7} {'p_raw':>11} {'p_BH':>8} {'sig':>4}")
print(f"  {'-'*68}")
for k in test_names:
    v = partial_tests[k]
    sig = '**' if v['reject_bh'] else ('*' if v['p'] < 0.05 else ' ')
    print(f"  {k:<38} {v['rho']:+7.3f} {v['p']:11.4e} {v['p_adj_bh']:8.4f} {sig:>4}")

n_bh_reject = sum(1 for k in test_names if partial_tests[k]['reject_bh'])
print(f"\n  BH-FDR: {n_bh_reject}/{len(test_names)} tests survive correction")


# ================================================================
# 7. REGRESSIONS
# ================================================================
print("\n[7] OLS regressions...")

# r1 ~ log(ξ/R) + log(N) + log(dR/R) + Inc + eD
reg_full, r2_full = ols_regression(
    np.column_stack([log_xiRe, log_N, log_dRRe, Inc, eD]), r1,
    ['log_xiRe', 'log_N', 'log_dRRe', 'Inc', 'eD'])
print(f"  r1 ~ log(ξ/R) + log(N) + log(dR/R) + Inc + eD:  R²={r2_full}")
for cn in ['log_xiRe', 'log_N', 'log_dRRe', 'Inc', 'eD']:
    c = reg_full[cn]
    print(f"    {cn:<12}: β={c['beta']:+.4f} ± {c['se']:.4f}, p={c['p']:.4f}")

# AR1_grid ~ log(ξ/R) + log(N) + log(dR/R) + Inc + eD
if n_grid_valid >= 20:
    gv = ar1g_valid
    reg_grid, r2_grid = ols_regression(
        np.column_stack([log_xiRe[gv], log_N[gv], log_dRRe[gv], Inc[gv], eD[gv]]),
        ar1g[gv], ['log_xiRe', 'log_N', 'log_dRRe', 'Inc', 'eD'])
    print(f"\n  AR1g ~ log(ξ/R) + log(N) + log(dR/R) + Inc + eD:  R²={r2_grid}")
    for cn in ['log_xiRe', 'log_N', 'log_dRRe', 'Inc', 'eD']:
        c = reg_grid[cn]
        print(f"    {cn:<12}: β={c['beta']:+.4f} ± {c['se']:.4f}, p={c['p']:.4f}")
else:
    reg_grid, r2_grid = None, None


# ================================================================
# 8. DETRENDING SWEEP
# ================================================================
print(f"\n[8] Detrending sweep: partial ρ(r1 vs ξ/R | N, dR/R, Inc, eD)")
print(f"\n  {'mult':>6} {'<r1>':>8} {'ρ(part)':>8} {'p(part)':>12} {'ρ(AR1g)':>8} {'p(AR1g)':>12}")
print(f"  {'-'*62}")

sweep_partials = {}
for mult in DETREND_MULTS:
    r1_m = np.array([g[f'r1_{mult}'] for g in gals])
    rho_par, p_par = partial_spearman(r1_m, log_xiRe, log_N, log_dRRe, Inc, eD)

    ar1g_m = np.array([g[f'ar1g_{mult}'] for g in gals])
    vm = ~np.isnan(ar1g_m)
    if vm.sum() >= 20:
        rho_g, p_g = partial_spearman(ar1g_m[vm], log_xiRe[vm], log_N[vm],
                                       log_dRRe[vm], Inc[vm], eD[vm])
    else:
        rho_g, p_g = np.nan, np.nan

    sweep_partials[str(mult)] = {
        'mean_r1': round(float(np.mean(r1_m)), 4),
        'r1_partial_rho': rho_par, 'r1_partial_p': p_par,
        'ar1g_partial_rho': rho_g, 'ar1g_partial_p': p_g,
    }
    sig1 = '*' if p_par < 0.05 else ' '
    sig2 = '*' if (not np.isnan(p_g) and p_g < 0.05) else ' '
    rg_str = f'{rho_g:+8.3f}' if not np.isnan(rho_g) else '     N/A'
    pg_str = f'{p_g:12.4e}' if not np.isnan(p_g) else '         N/A'
    print(f"  {mult:6.2f} {np.mean(r1_m):8.4f} {rho_par:+8.3f} {p_par:12.4e}{sig1} "
          f"{rg_str} {pg_str}{sig2}")

n_detrend_sig_r1 = sum(1 for sv in sweep_partials.values() if sv['r1_partial_p'] < 0.05)
n_detrend_sig_g  = sum(1 for sv in sweep_partials.values()
                       if not np.isnan(sv['ar1g_partial_p']) and sv['ar1g_partial_p'] < 0.05)
print(f"\n  r1 partial sig:     {n_detrend_sig_r1}/{len(DETREND_MULTS)}")
print(f"  AR1_grid partial sig: {n_detrend_sig_g}/{len(DETREND_MULTS)}")


# ================================================================
# 9. PERMUTATION + BOOTSTRAP
# ================================================================
print(f"\n[9] Permutation ({N_PERM}) + bootstrap ({N_BOOT})...")

rng = np.random.default_rng(101)
obs_rho = corrs.get('r1_vs_xi_Re', {}).get('rho', np.nan)

perm_rho = np.zeros(N_PERM)
for p in range(N_PERM):
    idx = rng.permutation(nv)
    perm_rho[p] = stats.spearmanr(r1, xi_Re[idx])[0]
pp = float(np.mean(np.abs(perm_rho) >= abs(obs_rho)))
print(f"  r1 vs ξ/R bivariate: ρ={obs_rho:+.3f}, perm p={pp:.4f}")

brng = np.random.default_rng(303)
boot_rho = np.zeros(N_BOOT)
for b in range(N_BOOT):
    idx = brng.integers(0, nv, size=nv)
    boot_rho[b] = stats.spearmanr(r1[idx], xi_Re[idx])[0]
ci = np.percentile(boot_rho, [2.5, 97.5])
print(f"  Bootstrap ρ CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# AR1_grid permutation
if n_grid_valid >= 20:
    obs_rho_g = float(stats.spearmanr(ar1g[ar1g_valid], xi_Re[ar1g_valid])[0])
    perm_rho_g = np.zeros(N_PERM)
    nv_g = int(ar1g_valid.sum())
    for p_i in range(N_PERM):
        idx = rng.permutation(nv_g)
        perm_rho_g[p_i] = stats.spearmanr(ar1g[ar1g_valid], xi_Re[ar1g_valid][idx])[0]
    pp_g = float(np.mean(np.abs(perm_rho_g) >= abs(obs_rho_g)))
    print(f"  AR1_grid vs ξ/R bivariate: ρ={obs_rho_g:+.3f}, perm p={pp_g:.4f}")
else:
    obs_rho_g, pp_g = np.nan, np.nan


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Galaxies: {nv} (AR1_grid valid: {n_grid_valid})")

print(f"\n  --- Key Bivariate (Spearman) ---")
for key in ['r1_vs_xi_Re', 'AR1_vs_xi_Re', 'AR1_grid_vs_xi_Re',
            'r1_vs_Npts', 'r1_vs_dR_Re', 'r1_vs_Inc', 'r1_vs_eD']:
    if key in corrs:
        c = corrs[key]
        print(f"    {key:<28}: ρ={c['rho']:+.3f}, p={c['p']:.4e}")

print(f"\n  --- Partial Correlations (with BH-FDR) ---")
for k in test_names:
    v = partial_tests[k]
    sig = '**' if v['reject_bh'] else ('*' if v['p'] < 0.05 else '  ')
    print(f"    {k:<38}: ρ={v['rho']:+.3f}, p_raw={v['p']:.4e}, p_BH={v['p_adj_bh']:.4f} {sig}")

print(f"\n  BH-FDR: {n_bh_reject}/{len(test_names)} survive")

print(f"\n  --- Detrending Sweep ---")
print(f"    r1 partial sig:      {n_detrend_sig_r1}/{len(DETREND_MULTS)}")
print(f"    AR1_grid partial sig: {n_detrend_sig_g}/{len(DETREND_MULTS)}")

print(f"\n  --- Regression: r1 ~ log(ξ/R) + log(N) + log(dR/R) + Inc + eD ---")
c = reg_full['log_xiRe']
print(f"    b(ξ/R)={c['beta']:+.4f} ± {c['se']:.4f}, p={c['p']:.4f}")

if reg_grid:
    c = reg_grid['log_xiRe']
    print(f"  --- Regression: AR1g ~ same ---")
    print(f"    b(ξ/R)={c['beta']:+.4f} ± {c['se']:.4f}, p={c['p']:.4f}")

# Verdict
# Primary: r1 vs ξ/R | N,dRRe,Inc,eD
primary_partial = partial_tests.get('r1_vs_xiRe | N,dRRe,Inc,eD', {})
primary_sig = primary_partial.get('p', 1) < 0.05
primary_bh  = primary_partial.get('reject_bh', False)
detrend_robust = n_detrend_sig_r1 >= len(DETREND_MULTS) // 2

# AR1_grid partial (the N-independent metric)
grid_partial = partial_tests.get('AR1g_vs_xiRe | N,dRRe,Inc,eD', {})
grid_sig = grid_partial.get('p', 1) < 0.05

if primary_bh and detrend_robust:
    verdict = "XI_SCALING_SURVIVES_FDR"
    print(f"\n  VERDICT: {verdict}")
    print("    ξ/R survives all controls + BH-FDR + detrending sweep.")
elif primary_sig and detrend_robust:
    verdict = "XI_SCALING_SURVIVES_NOMINAL"
    print(f"\n  VERDICT: {verdict}")
    print("    ξ/R survives all controls at nominal p<0.05, robust to detrending.")
    print("    Does NOT survive BH-FDR — treat as suggestive.")
elif primary_sig:
    verdict = "XI_SCALING_FRAGILE"
    print(f"\n  VERDICT: {verdict}")
    print("    ξ/R survives controls nominally but not robust to detrending.")
else:
    verdict = "NO_XI_SCALING"
    print(f"\n  VERDICT: {verdict}")

if grid_sig:
    print(f"    AR1_grid (N-independent) also confirms: ρ={grid_partial['rho']:+.3f}, p={grid_partial['p']:.4e}")

# ---------- save ----------
per_gal = []
for g in gals:
    per_gal.append({
        'name': g['nm'], 'n': g['n'],
        'r1': round(float(g[f'r1_{PRIMARY}']), 4),
        'ar1': round(float(g[f'ar1_{PRIMARY}']), 4) if not np.isnan(g[f'ar1_{PRIMARY}']) else None,
        'ar1g': round(float(g[f'ar1g_{PRIMARY}']), 4) if not np.isnan(g[f'ar1g_{PRIMARY}']) else None,
        'xi_kpc': round(g['xi'], 3), 'R_ext': round(g['Re'], 2),
        'dR': round(g['dR'], 3), 'dR_Re': round(g['dR']/g['Re'], 4),
        'xi_Re': round(g['xi']/g['Re'], 4),
        'Vflat': round(g['Vf'], 1), 'Mb': round(g['Mb'], 0),
        'Inc': round(g['Inc'], 1), 'eD': round(g['eD'], 2),
        'env': g['env'],
    })

results = {
    'test': 'healing_length_acf_v3',
    'description': 'ACF (r1, AR1, AR1_grid) vs ξ/R. Controls: N, dR/R, Inc, eD. BH-FDR. Detrend sweep.',
    'parameters': {'min_pts': MIN_POINTS, 'ML': Msun_L36, 'helium': HELIUM,
                   'g_dag_SI': g_dagger_SI, 'n_perm': N_PERM, 'n_boot': N_BOOT,
                   'detrend_mults': DETREND_MULTS, 'primary_mult': PRIMARY,
                   'grid_step_kpc': GRID_STEP, 'min_grid_pts': MIN_GRID_PTS},
    'sample': {'n_galaxies': nv, 'n_grid_valid': n_grid_valid},
    'correlations': corrs,
    'partial_correlations': partial_tests,
    'n_bh_reject': n_bh_reject,
    'regression_r1_full': {'coefficients': reg_full, 'r_sq': r2_full},
    'regression_ar1g_full': {'coefficients': reg_grid, 'r_sq': r2_grid} if reg_grid else None,
    'detrending_sweep': sweep_partials,
    'n_detrend_sig_r1': n_detrend_sig_r1,
    'n_detrend_sig_ar1g': n_detrend_sig_g,
    'bootstrap': {'rho_ci95': [round(float(ci[0]),4), round(float(ci[1]),4)]},
    'permutation': {
        'r1_xiRe_perm_p': pp,
        'ar1g_xiRe_perm_p': pp_g if not np.isnan(pp_g) else None,
    },
    'verdict': verdict,
    'per_galaxy': per_gal,
}

out = os.path.join(RESULTS_DIR, 'summary_healing_length_acf.json')
with open(out, 'w') as f: json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
print("=" * 72)
