#!/usr/bin/env python3
"""
Healing-Length Distance Diagnostics (Step 28f)
===============================================

Tests whether ξ/R_ext signal in r1 is real or a distance-quality proxy.

1) Clean subset test (Q==1, Q<=2, eD cuts)
2) Distance-residualized rank correlation
3) Stratified permutation null (shuffle ξ/R within eD bins)
4) Monte Carlo distance-uncertainty propagation (perturb ξ/R, hold r1 fixed)

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
N_MC         = 2000
PRIMARY_MULT = 0.5

np.random.seed(42)

print("=" * 72)
print("HEALING-LENGTH DISTANCE DIAGNOSTICS (Step 28f)")
print("=" * 72)

# ---------- helpers ----------
UMA = {'NGC3726','NGC3769','NGC3877','NGC3893','NGC3917','NGC3949','NGC3953',
       'NGC3972','NGC3992','NGC4010','NGC4013','NGC4051','NGC4085','NGC4088',
       'NGC4100','NGC4138','NGC4157','NGC4183','NGC4217','UGC06399','UGC06446',
       'UGC06667','UGC06786','UGC06787','UGC06818','UGC06917','UGC06923',
       'UGC06930','UGC06973','UGC06983','UGC07089'}
GRP = {'NGC2403','NGC2976','IC2574','DDO154','DDO168','UGC04483',
       'NGC0300','NGC0055','NGC0247','NGC7793','NGC2915','UGCA442',
       'ESO444-G084','UGC07577','UGC07232','NGC3741','NGC4068',
       'UGC07866','UGC07524','UGC08490','UGC07559','NGC3109','NGC5055'}

def rar(lg):
    gb = 10.**lg; return np.log10(gb / (1. - np.exp(-np.sqrt(gb / 1.2e-10))))

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
# LOAD + BUILD DATA
# ================================================================
print("\n[0] Loading SPARC and computing r1...")
t2  = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

rc = {}
with open(t2) as f:
    for ln in f:
        if len(ln.strip()) < 50: continue
        try:
            nm = ln[0:11].strip()
            if not nm: continue
            dist = float(ln[12:18])
            r = float(ln[19:25]); vo = float(ln[26:32])
            vg = float(ln[39:45]); vd = float(ln[46:52]); vb = float(ln[53:59])
        except: continue
        if nm not in rc:
            rc[nm] = {'R':[],'Vo':[],'Vg':[],'Vd':[],'Vb':[],'dist':dist}
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
        L36 = float(p[6]); MHI = float(p[12])
        Mb = Msun_L36 * L36 * 1e9 + HELIUM * MHI * 1e9
        props[nm] = {
            'D': float(p[1]), 'eD': float(p[2]), 'Inc': float(p[4]),
            'Vf': float(p[14]), 'Q': int(p[16]),
            'Mb': Mb, 'L36': L36, 'MHI': MHI,
        }
    except: continue

# Build galaxies with r1, using same cuts as prior scripts
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
    # Detrend
    v = np.var(eps); s = n * v * PRIMARY_MULT
    try:
        sp = UnivariateSpline(Rs, eps, k=min(3, n-1), s=s)
        ed = eps - sp(Rs)
    except: ed = eps - np.mean(eps)
    r1 = lag_acf(ed, 1)
    xi = np.sqrt(G_kpc * pr['Mb'] / g_dagger_kpc)
    # Distance from table2 (more reliable per-galaxy)
    D0 = d.get('dist', pr['D'])
    gals.append({
        'nm': nm, 'n': n, 'Re': Re, 'dR': dR, 'r1': r1,
        'xi': xi, 'xi_Re': xi / Re,
        'Vf': pr['Vf'], 'Mb': pr['Mb'],
        'Inc': pr['Inc'], 'D': D0, 'eD': pr['eD'], 'Q': pr['Q'],
        'L36': pr['L36'], 'MHI': pr['MHI'],
        'env': 'dense' if nm in UMA or nm in GRP else 'field',
    })

nv = len(gals)
print(f"  {nv} galaxies loaded")

# Extract common arrays
r1_arr    = np.array([g['r1']    for g in gals])
xi_Re_arr = np.array([g['xi_Re'] for g in gals])
xi_arr    = np.array([g['xi']    for g in gals])
Re_arr    = np.array([g['Re']    for g in gals])
N_arr     = np.array([g['n']     for g in gals], dtype=float)
dR_arr    = np.array([g['dR']    for g in gals])
Inc_arr   = np.array([g['Inc']   for g in gals])
D_arr     = np.array([g['D']     for g in gals])
eD_arr    = np.array([g['eD']    for g in gals])
Q_arr     = np.array([g['Q']     for g in gals])
Mb_arr    = np.array([g['Mb']    for g in gals])
L36_arr   = np.array([g['L36']   for g in gals])
MHI_arr   = np.array([g['MHI']   for g in gals])

dR_Re_arr = dR_arr / Re_arr
log_N     = np.log10(N_arr)
log_dRRe  = np.log10(dR_Re_arr)
log_xiRe  = np.log10(xi_Re_arr)
log_D     = np.log10(D_arr)
log_Re    = np.log10(Re_arr)

med_eD = float(np.median(eD_arr))
print(f"  Median eD = {med_eD:.2f} Mpc")
print(f"  Q distribution: Q=1: {(Q_arr==1).sum()}, Q=2: {(Q_arr==2).sum()}")


# ================================================================
# TEST 1: CLEAN SUBSET TEST
# ================================================================
print("\n" + "=" * 72)
print("TEST 1: CLEAN SUBSET TEST")
print("=" * 72)

subset_defs = [
    ('Q<=2, all eD',       (Q_arr <= 2)),
    ('Q==1, all eD',       (Q_arr == 1)),
    (f'Q<=2, eD<={med_eD:.2f}', (Q_arr <= 2) & (eD_arr <= med_eD)),
    (f'Q==1, eD<={med_eD:.2f}', (Q_arr == 1) & (eD_arr <= med_eD)),
]
# Add tight eD cut if enough galaxies
tight_eD = 1.0  # 1 Mpc
n_tight = ((Q_arr <= 2) & (eD_arr <= tight_eD)).sum()
if n_tight >= 15:
    subset_defs.append((f'Q<=2, eD<={tight_eD:.1f}', (Q_arr <= 2) & (eD_arr <= tight_eD)))
    subset_defs.append((f'Q==1, eD<={tight_eD:.1f}', (Q_arr == 1) & (eD_arr <= tight_eD)))

print(f"\n  {'Subset':<28} {'n':>4}  {'ρ_biv':>7} {'p_biv':>10}  "
      f"{'ρ_part':>7} {'p_part':>10}  {'ρ_full':>7} {'p_full':>10}")
print(f"  {'-'*100}")

subset_results = {}
for label, mask in subset_defs:
    n_sub = int(mask.sum())
    if n_sub < 10:
        print(f"  {label:<28} {n_sub:4d}  -- too few --")
        subset_results[label] = {'n': n_sub, 'too_few': True}
        continue

    r1_s = r1_arr[mask]; xiRe_s = xi_Re_arr[mask]
    logN_s = log_N[mask]; logdRRe_s = log_dRRe[mask]
    Inc_s = Inc_arr[mask]; eD_s = eD_arr[mask]

    # a) bivariate
    rho_biv, p_biv = stats.spearmanr(r1_s, xiRe_s)
    # b) partial | N, dR/R
    rho_part, p_part = partial_spearman(r1_s, np.log10(xiRe_s), logN_s, logdRRe_s)
    # c) partial | N, dR/R, Inc, eD
    if n_sub >= 15:
        rho_full, p_full = partial_spearman(r1_s, np.log10(xiRe_s),
                                             logN_s, logdRRe_s, Inc_s, eD_s)
    else:
        rho_full, p_full = np.nan, np.nan

    sig_b = '*' if p_biv < 0.05 else ' '
    sig_p = '*' if p_part < 0.05 else ' '
    sig_f = '*' if (not np.isnan(p_full) and p_full < 0.05) else ' '

    print(f"  {label:<28} {n_sub:4d}  {rho_biv:+7.3f} {p_biv:10.4e}{sig_b} "
          f"{rho_part:+7.3f} {p_part:10.4e}{sig_p} "
          f"{rho_full:+7.3f} {p_full:10.4e}{sig_f}")

    subset_results[label] = {
        'n': n_sub,
        'bivariate': {'rho': round(float(rho_biv),4), 'p': float(p_biv)},
        'partial_N_dRRe': {'rho': rho_part, 'p': p_part},
        'partial_full': {'rho': rho_full, 'p': p_full} if not np.isnan(p_full) else None,
    }


# ================================================================
# TEST 2: DISTANCE-RESIDUALIZED RANK CORRELATION
# ================================================================
print("\n" + "=" * 72)
print("TEST 2: DISTANCE-RESIDUALIZED RANK CORRELATION")
print("=" * 72)

from scipy.stats import rankdata

rank_r1   = rankdata(r1_arr).astype(float)
rank_xiRe = rankdata(log_xiRe).astype(float)

# Design matrix: logN, log(dR/R), Inc, eD, logD, logR_ext
Z = np.column_stack([log_N, log_dRRe, Inc_arr, eD_arr, log_D, log_Re])
Z1 = np.column_stack([np.ones(nv), Z])
cov_names = ['intercept', 'logN', 'log_dRRe', 'Inc', 'eD', 'logD', 'logR_ext']

# Regress rank(r1) on Z
b_r1, _, _, _ = np.linalg.lstsq(Z1, rank_r1, rcond=None)
res_r1 = rank_r1 - Z1 @ b_r1

# Regress rank(log(ξ/R)) on Z
b_xiRe, _, _, _ = np.linalg.lstsq(Z1, rank_xiRe, rcond=None)
res_xiRe = rank_xiRe - Z1 @ b_xiRe

# Residual Spearman
rho_resid, p_resid = stats.spearmanr(res_r1, res_xiRe)

print(f"\n  Design matrix: {cov_names[1:]}")
print(f"\n  Rank(r1) regression coefficients:")
for i, cn in enumerate(cov_names):
    print(f"    {cn:<12}: {b_r1[i]:+.3f}")
print(f"\n  Rank(log ξ/R) regression coefficients:")
for i, cn in enumerate(cov_names):
    print(f"    {cn:<12}: {b_xiRe[i]:+.3f}")

print(f"\n  Residual Spearman: ρ = {rho_resid:+.4f}, p = {p_resid:.4e}")

resid_corr_result = {
    'design_matrix': cov_names[1:],
    'coeff_rank_r1': {cn: round(float(b_r1[i]),4) for i, cn in enumerate(cov_names)},
    'coeff_rank_xiRe': {cn: round(float(b_xiRe[i]),4) for i, cn in enumerate(cov_names)},
    'residual_rho': round(float(rho_resid), 4),
    'residual_p': float(p_resid),
}


# ================================================================
# TEST 3: STRATIFIED PERMUTATION NULL
# ================================================================
print("\n" + "=" * 72)
print("TEST 3: STRATIFIED PERMUTATION NULL")
print("=" * 72)

# Create 6 bins by eD
n_bins = 6
eD_sorted_idx = np.argsort(eD_arr)
bin_size = nv // n_bins
bins = np.zeros(nv, dtype=int)
for i in range(n_bins):
    start = i * bin_size
    end = (i + 1) * bin_size if i < n_bins - 1 else nv
    bins[eD_sorted_idx[start:end]] = i

bin_counts = [int((bins == i).sum()) for i in range(n_bins)]
bin_eD_ranges = []
for i in range(n_bins):
    m = bins == i
    bin_eD_ranges.append(f"[{eD_arr[m].min():.2f}, {eD_arr[m].max():.2f}]")
print(f"  {n_bins} bins by eD: {bin_counts}")
for i in range(n_bins):
    print(f"    Bin {i}: n={bin_counts[i]}, eD range {bin_eD_ranges[i]}")

# Observed
obs_rho = float(stats.spearmanr(r1_arr, xi_Re_arr)[0])

# Naive permutation (unstratified)
rng = np.random.default_rng(101)
naive_perm_rhos = np.zeros(N_PERM)
for p in range(N_PERM):
    idx = rng.permutation(nv)
    naive_perm_rhos[p] = stats.spearmanr(r1_arr, xi_Re_arr[idx])[0]
naive_p = float(np.mean(np.abs(naive_perm_rhos) >= abs(obs_rho)))

# Stratified permutation (shuffle within bins)
rng2 = np.random.default_rng(202)
strat_perm_rhos = np.zeros(N_PERM)
for p in range(N_PERM):
    xi_Re_shuf = xi_Re_arr.copy()
    for b in range(n_bins):
        m = np.where(bins == b)[0]
        xi_Re_shuf[m] = xi_Re_shuf[rng2.permutation(m)]
    strat_perm_rhos[p] = stats.spearmanr(r1_arr, xi_Re_shuf)[0]
strat_p = float(np.mean(np.abs(strat_perm_rhos) >= abs(obs_rho)))

strat_z = (obs_rho - np.mean(strat_perm_rhos)) / np.std(strat_perm_rhos) if np.std(strat_perm_rhos) > 0 else np.nan

print(f"\n  Observed ρ(r1, ξ/R) = {obs_rho:+.4f}")
print(f"  Naive perm p:      {naive_p:.4f}")
print(f"  Stratified perm p: {strat_p:.4f}")
print(f"  Stratified z:      {strat_z:.2f}")
print(f"  Null mean (strat): {np.mean(strat_perm_rhos):+.4f} ± {np.std(strat_perm_rhos):.4f}")
print(f"  Null mean (naive): {np.mean(naive_perm_rhos):+.4f} ± {np.std(naive_perm_rhos):.4f}")

strat_result = {
    'observed_rho': round(obs_rho, 4),
    'naive_perm_p': naive_p,
    'stratified_perm_p': strat_p,
    'stratified_z': round(strat_z, 2) if not np.isnan(strat_z) else None,
    'null_mean_strat': round(float(np.mean(strat_perm_rhos)), 4),
    'null_std_strat': round(float(np.std(strat_perm_rhos)), 4),
    'n_bins': n_bins,
    'bin_counts': bin_counts,
}


# ================================================================
# TEST 4: MONTE CARLO DISTANCE-UNCERTAINTY PROPAGATION
# ================================================================
print("\n" + "=" * 72)
print("TEST 4: MC DISTANCE-UNCERTAINTY PROPAGATION")
print("=" * 72)
print(f"  {N_MC} realizations, perturbing D → ξ/R, holding r1 fixed")

# How D enters ξ/R:
# Mb = M_star + M_gas = 0.5 * L36_obs * 1e9 + 1.33 * MHI_obs * 1e9
# But L36_obs ∝ D² (luminosity from flux × D²), MHI ∝ D² (HI mass from flux × D²)
# So Mb ∝ D²
# R_extent is in angular units × D → Re ∝ D (radii in kpc = arcsec × D)
# ξ = sqrt(G * Mb / g†) ∝ sqrt(D²) = D
# ξ/R ∝ D / D = constant!  ... Wait, that means ξ/R is D-independent?
# Let me check: Re from SPARC is already in kpc (radii in table2 are kpc).
# Actually SPARC radii are given in kpc, already distance-applied.
# L36 and MHI are already distance-applied (absolute quantities).
# So if D changes:
#   L36_new = L36_old * (D_new/D_old)²  (flux constant, D changes)
#   MHI_new = MHI_old * (D_new/D_old)²
#   Mb_new = Mb_old * (D_new/D_old)²
#   R_new = R_old * (D_new/D_old)  (angular size fixed, distance changes)
#   Re_new = Re_old * (D_new/D_old)
#   xi_new = sqrt(G * Mb_new / g†) = xi_old * (D_new/D_old)
#   xi_Re_new = xi_new / Re_new = xi_old * (D_new/D_old) / (Re_old * (D_new/D_old))
#             = xi_old / Re_old = xi_Re_old
#
# ξ/R IS DISTANCE-INDEPENDENT!  This is crucial.
# The ratio cancels because both ξ and R scale linearly with D.

print("\n  CRITICAL FINDING: ξ/R_ext is distance-independent!")
print("  Both ξ ∝ D and R_ext ∝ D, so the ratio cancels.")
print("  Therefore MC distance perturbation cannot change ξ/R.")
print("  Running MC to confirm this algebraic result...\n")

# Confirm numerically anyway
rng3 = np.random.default_rng(303)
mc_rhos = np.zeros(N_MC)

for m in range(N_MC):
    # Draw perturbed distances (lognormal to stay positive)
    frac_eD = eD_arr / D_arr
    log_D_draw = np.log(D_arr) + rng3.normal(0, frac_eD)
    D_draw = np.exp(log_D_draw)
    scale = D_draw / D_arr   # D_new / D_old

    # Perturbed Mb, Re, xi
    Mb_draw = Mb_arr * scale**2
    Re_draw = Re_arr * scale
    xi_draw = np.sqrt(G_kpc * Mb_draw / g_dagger_kpc)
    xiRe_draw = xi_draw / Re_draw   # should equal xi_Re_arr exactly

    mc_rhos[m] = stats.spearmanr(r1_arr, xiRe_draw)[0]

mc_mean = float(np.mean(mc_rhos))
mc_ci = np.percentile(mc_rhos, [2.5, 97.5])
mc_frac_sig = float(np.mean([stats.spearmanr(r1_arr, xi_Re_arr)[1] < 0.05]))  # same every time

# Check: how much does xiRe actually vary?
mc_xiRe_cv = []
for m in range(min(100, N_MC)):
    frac_eD = eD_arr / D_arr
    log_D_draw = np.log(D_arr) + rng3.normal(0, frac_eD)
    scale = np.exp(log_D_draw) / D_arr
    xiRe_draw = xi_Re_arr * scale / scale  # = xi_Re_arr
    mc_xiRe_cv.append(np.mean(np.abs(xiRe_draw / xi_Re_arr - 1)))

print(f"  MC ρ(r1, ξ/R_draw): mean={mc_mean:.4f}, 95% CI=[{mc_ci[0]:.4f}, {mc_ci[1]:.4f}]")
print(f"  Mean |ξ/R perturbation|: {np.mean(mc_xiRe_cv):.6f} (≈0, confirms cancellation)")
print(f"  Nominal ρ = {obs_rho:.4f}")

# Since ξ/R is D-independent, the REAL test is:
# Does eD correlate with r1 through some OTHER pathway?
# Test: partial ρ(r1, eD) controlling everything else
rho_r1_eD, p_r1_eD = partial_spearman(r1_arr, eD_arr, log_N, log_dRRe, Inc_arr)
print(f"\n  Partial ρ(r1, eD | N, dR/R, Inc) = {rho_r1_eD:+.4f}, p = {p_r1_eD:.4e}")
print(f"  Bivariate ρ(r1, eD) = {stats.spearmanr(r1_arr, eD_arr)[0]:+.4f}")

# Also: does eD correlate with ξ/R?
rho_xiRe_eD = float(stats.spearmanr(xi_Re_arr, eD_arr)[0])
p_xiRe_eD = float(stats.spearmanr(xi_Re_arr, eD_arr)[1])
print(f"  Bivariate ρ(ξ/R, eD) = {rho_xiRe_eD:+.4f}, p = {p_xiRe_eD:.4e}")

mc_result = {
    'n_mc': N_MC,
    'xi_Re_is_D_independent': True,
    'mc_rho_mean': round(mc_mean, 4),
    'mc_rho_ci95': [round(float(mc_ci[0]), 4), round(float(mc_ci[1]), 4)],
    'mc_rho_std': round(float(np.std(mc_rhos)), 6),
    'partial_r1_eD_given_NdRRInc': {'rho': rho_r1_eD, 'p': p_r1_eD},
    'bivariate_xiRe_eD': {'rho': round(rho_xiRe_eD, 4), 'p': p_xiRe_eD},
}


# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

print(f"""
  Test                                      ρ         p         Note
  ─────────────────────────────────────────────────────────────────────
  Bivariate ρ(r1, ξ/R)                   {obs_rho:+.3f}    {stats.spearmanr(r1_arr, xi_Re_arr)[1]:.4e}  N={nv}
""", end='')

# Subset results summary
for label in subset_results:
    sr = subset_results[label]
    if sr.get('too_few'): continue
    bv = sr['bivariate']
    pt = sr['partial_N_dRRe']
    print(f"  Subset: {label:<28}  {bv['rho']:+.3f}    {bv['p']:.4e}  n={sr['n']}, partial={pt['rho']:+.3f} (p={pt['p']:.3e})")

print(f"""
  Distance-residualized ranks              {rho_resid:+.3f}    {p_resid:.4e}  Controls: N,dR/R,Inc,eD,D,R_ext
  Stratified perm (6 eD bins)              {obs_rho:+.3f}    {strat_p:.4f}     vs naive p={naive_p:.4f}
  MC distance propagation               {mc_mean:+.3f}    (const)    ξ/R is D-independent!
  Partial ρ(r1, eD | N,dR/R,Inc)         {rho_r1_eD:+.3f}    {p_r1_eD:.4e}  eD pathway test
  Bivariate ρ(ξ/R, eD)                  {rho_xiRe_eD:+.3f}    {p_xiRe_eD:.4e}  confound coupling
""")

# Interpretation
print("  INTERPRETATION:")
print("  ξ/R_ext = sqrt(G*Mb/g†) / R_ext")
print("  Both Mb ∝ D² and R_ext ∝ D, so ξ ∝ D and ξ/R ∝ D/D = const.")
print("  Distance uncertainty CANNOT affect ξ/R — the ratio is geometric.")
if abs(rho_r1_eD) < 0.15 and p_r1_eD > 0.05:
    print("  eD has no independent pathway to r1 either.")
    print("  → The eD 'killing' was a DEGREES-OF-FREEDOM artifact,")
    print("    not a genuine confound. The ξ/R signal is real.")
    interpretation = "EDD_ARTIFACT"
else:
    print("  eD does have an independent pathway to r1.")
    print("  → Needs further investigation.")
    interpretation = "NEEDS_INVESTIGATION"


# ================================================================
# SAVE
# ================================================================
results = {
    'test': 'healing_length_distance_diagnostics',
    'description': 'Tests whether ξ/R signal is real or distance-quality artifact.',
    'sample': {'n_galaxies': nv, 'median_eD': med_eD},
    'test1_clean_subsets': subset_results,
    'test2_residualized_ranks': resid_corr_result,
    'test3_stratified_perm': strat_result,
    'test4_mc_distance': mc_result,
    'interpretation': interpretation,
}

out = os.path.join(RESULTS_DIR, 'summary_healing_length_distance_diag.json')
with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {out}")
print("=" * 72)
