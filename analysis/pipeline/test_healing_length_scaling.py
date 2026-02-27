#!/usr/bin/env python3
"""
Healing-Length Scaling Test (Step 28) — ξ vs Coherence Length
=============================================================

Tests whether the short-range coherence scale of detrended RAR residuals
scales with the BEC healing length ξ = sqrt(G*M_b/g†).

1) Per-galaxy ACF coherence length Lc_acf (detrended residuals)
2) Per-galaxy short-scale Lomb-Scargle λ_peak_short
3) Healing length ξ_kpc from baryonic mass
4) Cross-galaxy correlations: Lc vs ξ, λ vs ξ, controlling for R_extent
5) Permutation + shuffle null tests

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Constants
g_dagger_SI = 1.20e-10       # m/s^2
kpc_m = 3.086e19              # m per kpc
G_kpc = 4.30091e-6            # kpc (km/s)^2 / Msun
Msun_L36 = 0.5               # M/L ratio at 3.6 μm (Msun/Lsun)
HELIUM_FACTOR = 1.33          # M_gas = 1.33 * M_HI

# g† in (km/s)^2 / kpc:  g_SI * kpc_m / 1e6
g_dagger_kpc = g_dagger_SI * kpc_m / 1e6  # (km/s)^2 / kpc

MIN_POINTS = 15
N_PERM = 5000
N_BOOT = 2000
N_SHUF = 500   # within-galaxy shuffles for null

np.random.seed(42)

print("=" * 72)
print("HEALING-LENGTH SCALING TEST (Step 28)")
print("=" * 72)
print(f"  g† = {g_dagger_SI:.2e} m/s² = {g_dagger_kpc:.4f} (km/s)²/kpc")
print(f"  G = {G_kpc:.6e} kpc (km/s)²/Msun")
print(f"  M/L_3.6 = {Msun_L36}, helium factor = {HELIUM_FACTOR}")
print(f"  ξ = sqrt(G M_b / g†)")
print(f"  Min pts: {MIN_POINTS}, Perms: {N_PERM}, Boots: {N_BOOT}")


# ================================================================
# ENVIRONMENT
# ================================================================
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
    'NGC2403': 'M81', 'NGC2976': 'M81', 'IC2574': 'M81',
    'DDO154': 'M81', 'DDO168': 'M81', 'UGC04483': 'M81',
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor',
    'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    'NGC2915': 'CenA', 'UGCA442': 'CenA', 'ESO444-G084': 'CenA',
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI',
    'NGC4068': 'CVnI', 'UGC07866': 'CVnI', 'UGC07524': 'CVnI',
    'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    'NGC3109': 'Antlia', 'NGC5055': 'M101',
}

def classify_env(name):
    return 'dense' if name in UMA_GALAXIES or name in GROUP_MEMBERS else 'field'


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def lag_autocorrelation(x, lag=1):
    n = len(x)
    if n <= lag + 1:
        return np.nan
    xbar = np.mean(x)
    var = np.var(x, ddof=0)
    if var < 1e-30:
        return np.nan
    cov = np.mean((x[:n - lag] - xbar) * (x[lag:] - xbar))
    return cov / var


def safe_corr_pair(x, y):
    """Return Spearman/Pearson stats with graceful handling of small/degenerate samples."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, n
    rho_s, p_s = stats.spearmanr(x, y)
    rho_p, p_p = stats.pearsonr(x, y)
    return rho_s, p_s, rho_p, p_p, n


# ================================================================
# 1. LOAD SPARC DATA
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

rc_data = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50:
            continue
        try:
            name = line[0:11].strip()
            if not name:
                continue
            dist = float(line[12:18].strip())
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'Vgas': [],
                             'Vdisk': [], 'Vbul': [], 'dist': dist}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'Vgas', 'Vdisk', 'Vbul']:
        rc_data[name][key] = np.array(rc_data[name][key])

# Load galaxy properties + masses from MRT
sparc_props = {}
with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break
for line in mrt_lines[data_start:]:
    if not line.strip() or line.startswith('#'):
        continue
    try:
        name = line[0:11].strip()
        if not name:
            continue
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        L36 = float(parts[6])     # 10^9 Lsun
        MHI = float(parts[12])    # 10^9 Msun
        Rdisk = float(parts[10])  # kpc

        # Baryonic mass
        M_star = Msun_L36 * L36 * 1e9   # Msun
        M_gas = HELIUM_FACTOR * MHI * 1e9  # Msun
        M_b = M_star + M_gas

        sparc_props[name] = {
            'D': float(parts[1]),
            'Inc': float(parts[4]),
            'L36': L36,
            'MHI': MHI,
            'Rdisk': Rdisk,
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
            'M_star': M_star,
            'M_gas': M_gas,
            'M_b': M_b,
        }
    except (ValueError, IndexError):
        continue

print(f"  Loaded {len(rc_data)} galaxies with rotation curves")
print(f"  Loaded {len(sparc_props)} galaxies with properties + masses")


# ================================================================
# 2. BUILD PER-GALAXY DATA
# ================================================================
print("\n[2] Computing residuals, detrending, coherence lengths...")

galaxy_data = []
n_rejected = 0
n_missing_mass = 0

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue
    if prop['M_b'] <= 0:
        n_missing_mass += 1
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < MIN_POINTS:
        n_rejected += 1
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)

    n = len(residuals)
    R_extent = float(R_sorted[-1] - R_sorted[0])
    if R_extent <= 0:
        continue
    dR = float(np.median(np.diff(R_sorted)))

    # Spline detrending
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    # Healing length
    M_b = prop['M_b']
    xi_kpc = np.sqrt(G_kpc * M_b / g_dagger_kpc)

    med_log_gbar = float(np.median(log_gbar))

    galaxy_data.append({
        'name': name,
        'R': R_sorted,
        'eps_raw': residuals,
        'eps_det': eps_det,
        'n_pts': n,
        'dR': dR,
        'R_extent': R_extent,
        'Rdisk': prop['Rdisk'],
        'env': classify_env(name),
        'Vflat': prop['Vflat'],
        'med_log_gbar': med_log_gbar,
        'M_b': M_b,
        'M_star': prop['M_star'],
        'xi_kpc': xi_kpc,
    })

n_galaxies = len(galaxy_data)
print(f"  Galaxies with N >= {MIN_POINTS} and valid M_b: {n_galaxies}")
print(f"  Rejected (N < {MIN_POINTS}): {n_rejected}")
print(f"  Missing mass: {n_missing_mass}")


# ================================================================
# 3. COMPUTE Lc_acf PER GALAXY
# ================================================================
print("\n[3] Computing ACF coherence length Lc per galaxy...")

for g in galaxy_data:
    eps = g['eps_det']
    R = g['R']
    n = g['n_pts']
    dR = g['dR']

    K = min(10, n // 3)

    acf_vals = []
    sep_vals = []
    for k in range(1, K + 1):
        a = lag_autocorrelation(eps, lag=k)
        acf_vals.append(a)
        sep_vals.append(k * dR)

    acf_arr = np.array(acf_vals)
    sep_arr = np.array(sep_vals)

    g['acf1'] = float(acf_arr[0]) if len(acf_arr) > 0 and not np.isnan(acf_arr[0]) else np.nan
    g['acf2'] = float(acf_arr[1]) if len(acf_arr) > 1 and not np.isnan(acf_arr[1]) else np.nan

    # Method a: first sep where acf <= exp(-1)
    Lc = np.nan
    e_inv = np.exp(-1)
    finite_mask = ~np.isnan(acf_arr)

    if np.any(finite_mask):
        acf_f = acf_arr[finite_mask]
        sep_f = sep_arr[finite_mask]

        # a) first crossing below e^-1
        below = np.where(acf_f <= e_inv)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # Interpolate between previous and this point
                a0_v, a1_v = acf_f[idx - 1], acf_f[idx]
                s0_v, s1_v = sep_f[idx - 1], sep_f[idx]
                if a0_v != a1_v:
                    Lc = s0_v + (e_inv - a0_v) * (s1_v - s0_v) / (a1_v - a0_v)
                else:
                    Lc = s0_v
            else:
                Lc = sep_f[0]

        # b) exponential fit on positive acf values
        if np.isnan(Lc):
            pos_mask = acf_f > 0
            if np.sum(pos_mask) >= 2:
                log_acf = np.log(acf_f[pos_mask])
                sep_pos = sep_f[pos_mask]
                try:
                    slope, intercept, _, _, _ = stats.linregress(sep_pos, log_acf)
                    if slope < 0:
                        Lc = -1.0 / slope
                except Exception:
                    pass

        # c) first zero crossing
        if np.isnan(Lc):
            zeros = np.where(acf_f <= 0)[0]
            if len(zeros) > 0:
                idx = zeros[0]
                if idx > 0:
                    a0_v, a1_v = acf_f[idx - 1], acf_f[idx]
                    s0_v, s1_v = sep_f[idx - 1], sep_f[idx]
                    if a0_v != a1_v:
                        Lc = s0_v + (0 - a0_v) * (s1_v - s0_v) / (a1_v - a0_v)
                    else:
                        Lc = s0_v
                else:
                    Lc = sep_f[0]

    g['Lc_acf'] = float(Lc) if not np.isnan(Lc) else np.nan

# Filter galaxies with valid Lc
valid_gals = [g for g in galaxy_data if not np.isnan(g['Lc_acf']) and g['Lc_acf'] > 0]
n_valid_Lc = len(valid_gals)
print(f"  Valid Lc: {n_valid_Lc}/{n_galaxies}")
if n_valid_Lc < 5:
    raise RuntimeError(
        f"Too few galaxies with valid coherence length ({n_valid_Lc}) for correlation/regression tests."
    )
print(f"  Lc range: {min(g['Lc_acf'] for g in valid_gals):.2f} – "
      f"{max(g['Lc_acf'] for g in valid_gals):.2f} kpc")
print(f"  ξ range:  {min(g['xi_kpc'] for g in valid_gals):.2f} – "
      f"{max(g['xi_kpc'] for g in valid_gals):.2f} kpc")


# ================================================================
# 4. SHORT-SCALE LOMB-SCARGLE λ_peak_short
# ================================================================
print("\n[4] Short-scale Lomb-Scargle per galaxy...")

for g in galaxy_data:
    R = g['R']
    eps = g['eps_det']
    n = g['n_pts']
    dR = g['dR']
    R_ext = g['R_extent']

    std_eps = np.std(eps)
    if std_eps < 1e-30:
        g['lambda_short'] = np.nan
        g['power_short'] = np.nan
        g['lambda_all'] = np.nan
        continue

    y = (eps - np.mean(eps)) / std_eps

    # Full frequency grid
    f_min = 1.0 / R_ext
    f_max = (n / 2.0) / R_ext
    n_freq = min(500, 10 * n)
    freq_grid = np.linspace(f_min, f_max, n_freq)

    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)

    # Full peak
    idx_all = np.argmax(power)
    f_all = freq_grid[idx_all]
    g['lambda_all'] = float(1.0 / f_all)

    # Short-scale: restrict to wavelengths in [2*dR, 0.8*R_extent]
    wl_grid = 1.0 / freq_grid
    short_mask = (wl_grid >= 2 * dR) & (wl_grid <= 0.8 * R_ext)
    if np.any(short_mask):
        power_short = power.copy()
        power_short[~short_mask] = -1
        idx_short = np.argmax(power_short)
        f_short = freq_grid[idx_short]
        g['lambda_short'] = float(1.0 / f_short)
        g['power_short'] = float(power[idx_short])
    else:
        g['lambda_short'] = np.nan
        g['power_short'] = np.nan

n_valid_ls = sum(1 for g in galaxy_data if not np.isnan(g.get('lambda_short', np.nan)))
print(f"  Valid λ_short: {n_valid_ls}/{n_galaxies}")


# ================================================================
# 5. CROSS-GALAXY CORRELATIONS
# ================================================================
print("\n[5] Cross-galaxy correlations...")

# Build arrays from valid_gals (have Lc)
log_Lc = np.log10([g['Lc_acf'] for g in valid_gals])
log_xi = np.log10([g['xi_kpc'] for g in valid_gals])
log_Rext = np.log10([g['R_extent'] for g in valid_gals])
log_dR = np.log10([g['dR'] for g in valid_gals])
log_Npts = np.log10([g['n_pts'] for g in valid_gals])
Lc_arr = np.array([g['Lc_acf'] for g in valid_gals])
xi_arr = np.array([g['xi_kpc'] for g in valid_gals])
Rext_arr = np.array([g['R_extent'] for g in valid_gals])

# Lambda arrays (may have different valid set)
ls_valid = [g for g in galaxy_data if not np.isnan(g.get('lambda_short', np.nan))
            and g.get('lambda_short', 0) > 0 and g['xi_kpc'] > 0]
log_lam = np.log10([g['lambda_short'] for g in ls_valid])
log_xi_ls = np.log10([g['xi_kpc'] for g in ls_valid])
log_Rext_ls = np.log10([g['R_extent'] for g in ls_valid])

correlations = {}

# Spearman + Pearson for key pairs
pairs = [
    ('Lc_vs_xi', log_Lc, log_xi),
    ('Lc_vs_Rext', log_Lc, log_Rext),
    ('lambda_short_vs_xi', log_lam, log_xi_ls),
    ('lambda_short_vs_Rext', log_lam, log_Rext_ls),
]

for label, x, y_arr in pairs:
    rho_s, p_s, rho_p, p_p, n_pair = safe_corr_pair(x, y_arr)
    correlations[label] = {
        'spearman_rho': round(float(rho_s), 4) if np.isfinite(rho_s) else None,
        'spearman_p': float(p_s) if np.isfinite(p_s) else None,
        'pearson_r': round(float(rho_p), 4) if np.isfinite(rho_p) else None,
        'pearson_p': float(p_p) if np.isfinite(p_p) else None,
        'n': n_pair,
    }
    if np.isfinite(rho_s) and np.isfinite(rho_p):
        print(f"  {label:<28}: ρ_S={rho_s:+.3f} (p={p_s:.4f}), r_P={rho_p:+.3f} (p={p_p:.4f}), N={n_pair}")
    else:
        print(f"  {label:<28}: insufficient dynamic range/sample for stable correlation (N={n_pair})")

# Normalized: Lc/R_extent vs xi/R_extent
Lc_norm = Lc_arr / Rext_arr
xi_norm = xi_arr / Rext_arr
rho_s, p_s, rho_p, p_p, n_pair = safe_corr_pair(np.log10(Lc_norm), np.log10(xi_norm))
correlations['Lc_norm_vs_xi_norm'] = {
    'spearman_rho': round(float(rho_s), 4) if np.isfinite(rho_s) else None,
    'spearman_p': float(p_s) if np.isfinite(p_s) else None,
    'pearson_r': round(float(rho_p), 4) if np.isfinite(rho_p) else None,
    'pearson_p': float(p_p) if np.isfinite(p_p) else None,
    'n': n_pair,
}
if np.isfinite(rho_s) and np.isfinite(rho_p):
    print(f"  {'Lc/Rext vs xi/Rext':<28}: ρ_S={rho_s:+.3f} (p={p_s:.4f}), r_P={rho_p:+.3f} (p={p_p:.4f})")
else:
    print(f"  {'Lc/Rext vs xi/Rext':<28}: insufficient dynamic range/sample for stable correlation (N={n_pair})")


# ================================================================
# 6. LOG-LOG REGRESSION
# ================================================================
print("\n[6] Log-log regressions...")

# Simple: log(Lc) = a + b*log(ξ)
slope_xi, intercept_xi, r_val, p_val, se_slope = stats.linregress(log_xi, log_Lc)
print(f"  log(Lc) = {intercept_xi:.3f} + {slope_xi:.3f} * log(ξ)")
print(f"    slope = {slope_xi:.3f} ± {se_slope:.3f}, p = {p_val:.4e}")

regression_simple = {
    'slope': round(float(slope_xi), 4),
    'intercept': round(float(intercept_xi), 4),
    'slope_se': round(float(se_slope), 4),
    'r_squared': round(float(r_val**2), 4),
    'p_value': float(p_val),
}

# Multiple: log(Lc) = a + b*log(ξ) + c*log(R_ext) + d*log(dR) + e*log(N)
X = np.column_stack([log_xi, log_Rext, log_dR, log_Npts])
y = np.array(log_Lc)
# OLS via numpy
X_aug = np.column_stack([np.ones(len(y)), X])
try:
    beta, res, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard errors
    n_obs = len(y)
    n_params = X_aug.shape[1]
    dof = n_obs - n_params
    mse = ss_res / dof if dof > 0 else np.nan
    cov_beta = mse * np.linalg.pinv(X_aug.T @ X_aug)
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    # t-values and p-values
    t_vals = np.divide(beta, se_beta, out=np.full_like(beta, np.nan), where=se_beta > 0)
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=dof) if dof > 0 else np.full_like(beta, np.nan)

    coef_names = ['intercept', 'log_xi', 'log_Rext', 'log_dR', 'log_Npts']
    print(f"\n  Multiple regression: log(Lc) = a + b*log(ξ) + c*log(R_ext) + d*log(dR) + e*log(N)")
    print(f"  R² = {r2_multi:.4f}")
    for i, cn in enumerate(coef_names):
        print(f"    {cn:<12}: β={beta[i]:+.4f} ± {se_beta[i]:.4f}, t={t_vals[i]:+.2f}, p={p_vals[i]:.4f}")

    regression_multi = {
        'r_squared': round(float(r2_multi), 4),
        'coefficients': {
            cn: {
                'beta': round(float(beta[i]), 4),
                'se': round(float(se_beta[i]), 4),
                't': round(float(t_vals[i]), 2),
                'p': round(float(p_vals[i]), 4),
            }
            for i, cn in enumerate(coef_names)
        },
    }
except Exception as e:
    print(f"  Multiple regression failed: {e}")
    regression_multi = None


# ================================================================
# 7. PERMUTATION NULL: shuffle ξ labels
# ================================================================
print(f"\n[7] Permutation null (shuffle ξ labels, {N_PERM} perms)...")

perm_rng = np.random.default_rng(101)

obs_rho_Lc_xi = safe_corr_pair(log_Lc, log_xi)[0]
obs_rho_lam_xi = safe_corr_pair(log_lam, log_xi_ls)[0]

perm_p_Lc = np.nan
perm_p_lam = np.nan

if np.isfinite(obs_rho_Lc_xi):
    perm_rho_Lc = np.zeros(N_PERM)
    for p in range(N_PERM):
        shuf_xi = perm_rng.permutation(log_xi)
        perm_rho_Lc[p] = stats.spearmanr(log_Lc, shuf_xi)[0]
    perm_p_Lc = float(np.mean(np.abs(perm_rho_Lc) >= abs(obs_rho_Lc_xi)))

if np.isfinite(obs_rho_lam_xi):
    perm_rho_lam = np.zeros(N_PERM)
    for p in range(N_PERM):
        shuf_xi_ls = perm_rng.permutation(log_xi_ls)
        perm_rho_lam[p] = stats.spearmanr(log_lam, shuf_xi_ls)[0]
    perm_p_lam = float(np.mean(np.abs(perm_rho_lam) >= abs(obs_rho_lam_xi)))

print(f"  Lc vs ξ: observed ρ = {obs_rho_Lc_xi:+.4f}, perm p = {perm_p_Lc:.4f}")
print(f"  λ  vs ξ: observed ρ = {obs_rho_lam_xi:+.4f}, perm p = {perm_p_lam:.4f}")


# ================================================================
# 8. WITHIN-GALAXY SHUFFLE NULL
# ================================================================
print(f"\n[8] Within-galaxy shuffle null ({N_SHUF} shuffles)...")

shuf_rng = np.random.default_rng(202)

# For each galaxy, shuffle eps order, recompute Lc, then recompute correlations
shuf_rho_Lc_xi = np.zeros(N_SHUF)

for s in range(N_SHUF):
    shuf_log_Lc = []
    shuf_log_xi_list = []

    for g in valid_gals:
        eps = g['eps_det']
        R = g['R']
        n = g['n_pts']
        dR = g['dR']

        eps_shuf = shuf_rng.permutation(eps)
        K = min(10, n // 3)

        acf_vals = []
        for k in range(1, K + 1):
            acf_vals.append(lag_autocorrelation(eps_shuf, lag=k))

        acf_arr = np.array(acf_vals)
        sep_arr = np.array([k * dR for k in range(1, K + 1)])
        finite_mask = ~np.isnan(acf_arr)

        Lc_s = np.nan
        if np.any(finite_mask):
            acf_f = acf_arr[finite_mask]
            sep_f = sep_arr[finite_mask]
            e_inv = np.exp(-1)

            below = np.where(acf_f <= e_inv)[0]
            if len(below) > 0:
                idx = below[0]
                if idx > 0 and acf_f[idx - 1] != acf_f[idx]:
                    Lc_s = sep_f[idx - 1] + (e_inv - acf_f[idx - 1]) * (sep_f[idx] - sep_f[idx - 1]) / (acf_f[idx] - acf_f[idx - 1])
                else:
                    Lc_s = sep_f[idx] if len(sep_f) > idx else np.nan

            if np.isnan(Lc_s):
                pos_mask = acf_f > 0
                if np.sum(pos_mask) >= 2:
                    try:
                        sl, _, _, _, _ = stats.linregress(sep_f[pos_mask], np.log(acf_f[pos_mask]))
                        if sl < 0:
                            Lc_s = -1.0 / sl
                    except Exception:
                        pass

            if np.isnan(Lc_s):
                zeros = np.where(acf_f <= 0)[0]
                if len(zeros) > 0:
                    idx = zeros[0]
                    if idx > 0 and acf_f[idx - 1] != acf_f[idx]:
                        Lc_s = sep_f[idx - 1] + (0 - acf_f[idx - 1]) * (sep_f[idx] - sep_f[idx - 1]) / (acf_f[idx] - acf_f[idx - 1])
                    else:
                        Lc_s = sep_f[0]

        if not np.isnan(Lc_s) and Lc_s > 0:
            shuf_log_Lc.append(np.log10(Lc_s))
            shuf_log_xi_list.append(np.log10(g['xi_kpc']))

    if len(shuf_log_Lc) >= 10:
        shuf_rho_Lc_xi[s] = stats.spearmanr(shuf_log_Lc, shuf_log_xi_list)[0]
    else:
        shuf_rho_Lc_xi[s] = np.nan

shuf_valid = shuf_rho_Lc_xi[~np.isnan(shuf_rho_Lc_xi)]
shuf_mean = float(np.mean(shuf_valid))
shuf_std = float(np.std(shuf_valid))
shuf_z = (obs_rho_Lc_xi - shuf_mean) / shuf_std if shuf_std > 0 else np.nan

print(f"  Observed Lc-ξ ρ:  {obs_rho_Lc_xi:+.4f}")
print(f"  Shuffled Lc-ξ ρ:  {shuf_mean:+.4f} ± {shuf_std:.4f}")
print(f"  Z-score:          {shuf_z:.2f}")


# ================================================================
# 9. BOOTSTRAP CI FOR SLOPE AND RHO
# ================================================================
print(f"\n[9] Bootstrap CI ({N_BOOT} resamples)...")

boot_rng = np.random.default_rng(303)
n_v = len(valid_gals)
boot_slopes = np.zeros(N_BOOT)
boot_rhos = np.zeros(N_BOOT)

for b in range(N_BOOT):
    idx = boot_rng.integers(0, n_v, size=n_v)
    bx = log_xi[idx]
    by = log_Lc[idx]
    boot_slopes[b] = stats.linregress(bx, by)[0]
    boot_rhos[b] = stats.spearmanr(bx, by)[0]

slope_ci = np.percentile(boot_slopes, [2.5, 97.5])
rho_ci = np.percentile(boot_rhos, [2.5, 97.5])

print(f"  Slope 95% CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
print(f"  ρ_S   95% CI: [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Galaxies: {n_valid_Lc} with valid Lc, {len(ls_valid)} with valid λ_short")
print(f"  ξ range: {min(g['xi_kpc'] for g in valid_gals):.1f} – {max(g['xi_kpc'] for g in valid_gals):.1f} kpc")
print(f"  Lc range: {min(g['Lc_acf'] for g in valid_gals):.1f} – {max(g['Lc_acf'] for g in valid_gals):.1f} kpc")

print(f"\n  --- Key Correlations ---")
for label in ['Lc_vs_xi', 'Lc_vs_Rext', 'lambda_short_vs_xi', 'lambda_short_vs_Rext', 'Lc_norm_vs_xi_norm']:
    c = correlations[label]
    rho = c['spearman_rho']
    p = c['spearman_p']
    if rho is None or p is None:
        print(f"    {label:<28}: unavailable (insufficient dynamic range/sample)")
    else:
        print(f"    {label:<28}: ρ = {rho:+.3f} (p = {p:.4e})")

print(f"\n  --- Simple Regression: log(Lc) = a + b*log(ξ) ---")
print(f"    slope b = {regression_simple['slope']:.3f} ± {regression_simple['slope_se']:.3f}")
print(f"    R² = {regression_simple['r_squared']:.3f}, p = {regression_simple['p_value']:.4e}")
print(f"    Bootstrap 95% CI for slope: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")

if regression_multi:
    b_xi = regression_multi['coefficients']['log_xi']
    b_Rext = regression_multi['coefficients']['log_Rext']
    print(f"\n  --- Multiple Regression (controlling for R_ext, dR, N) ---")
    print(f"    b(ξ)    = {b_xi['beta']:+.3f} ± {b_xi['se']:.3f}, p = {b_xi['p']:.4f}")
    print(f"    b(Rext) = {b_Rext['beta']:+.3f} ± {b_Rext['se']:.3f}, p = {b_Rext['p']:.4f}")
    print(f"    R² = {regression_multi['r_squared']:.3f}")

print(f"\n  --- Null Tests ---")
print(f"    Permutation (ξ-label shuffle): Lc-ξ p = {perm_p_Lc:.4f}, λ-ξ p = {perm_p_lam:.4f}")
print(f"    Within-galaxy shuffle: Lc-ξ ρ → {shuf_mean:+.3f} ± {shuf_std:.3f} (z = {shuf_z:.1f})")
print(f"    Bootstrap ρ CI: [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")

# Verdict
xi_p = correlations['Lc_vs_xi']['spearman_p']
norm_p = correlations['Lc_norm_vs_xi_norm']['spearman_p']
xi_significant = xi_p is not None and xi_p < 0.01
xi_survives = regression_multi and regression_multi['coefficients']['log_xi']['p'] < 0.05
norm_significant = norm_p is not None and norm_p < 0.05

if xi_significant and xi_survives and norm_significant:
    verdict = "STRONG_XI_SCALING"
    print(f"\n  VERDICT: {verdict}")
    print("    Lc scales with ξ, SURVIVES controlling for R_extent, normalized test passes.")
elif xi_significant and xi_survives:
    verdict = "XI_SCALING_PARTIAL"
    print(f"\n  VERDICT: {verdict}")
    print("    Lc scales with ξ and survives R_ext control, but normalized test marginal.")
elif xi_significant and not xi_survives:
    verdict = "CONFOUNDED_BY_SIZE"
    print(f"\n  VERDICT: {verdict}")
    print("    Lc correlates with ξ but does NOT survive R_extent control.")
    print("    Likely driven by shared galaxy-size scaling.")
elif not xi_significant:
    verdict = "NO_XI_SCALING"
    print(f"\n  VERDICT: {verdict}")
    print("    No significant Lc–ξ correlation detected.")
else:
    verdict = "AMBIGUOUS"
    print(f"\n  VERDICT: {verdict}")


# ================================================================
# SAVE JSON
# ================================================================
per_galaxy_out = []
for g in galaxy_data:
    entry = {
        'name': g['name'],
        'n_pts': g['n_pts'],
        'dR_kpc': round(g['dR'], 3),
        'R_extent_kpc': round(g['R_extent'], 2),
        'Rdisk_kpc': round(g['Rdisk'], 2),
        'Vflat': round(g['Vflat'], 1),
        'M_b_Msun': round(g['M_b'], 0),
        'xi_kpc': round(g['xi_kpc'], 3),
        'Lc_acf_kpc': round(g['Lc_acf'], 3) if not np.isnan(g.get('Lc_acf', np.nan)) else None,
        'acf1': round(g.get('acf1', np.nan), 4) if not np.isnan(g.get('acf1', np.nan)) else None,
        'acf2': round(g.get('acf2', np.nan), 4) if not np.isnan(g.get('acf2', np.nan)) else None,
        'lambda_short_kpc': round(g.get('lambda_short', np.nan), 3) if not np.isnan(g.get('lambda_short', np.nan)) else None,
        'power_short': round(g.get('power_short', np.nan), 4) if not np.isnan(g.get('power_short', np.nan)) else None,
        'lambda_all_kpc': round(g.get('lambda_all', np.nan), 3) if not np.isnan(g.get('lambda_all', np.nan)) else None,
        'env': g['env'],
        'med_log_gbar': round(g['med_log_gbar'], 3),
    }
    per_galaxy_out.append(entry)

results = {
    'test': 'healing_length_scaling',
    'description': ('Tests whether ACF coherence length Lc and short-scale spectral '
                     'wavelength λ scale with BEC healing length ξ = sqrt(G*M_b/g†).'),
    'parameters': {
        'min_points': MIN_POINTS,
        'ML_ratio_36': Msun_L36,
        'helium_factor': HELIUM_FACTOR,
        'g_dagger_SI': g_dagger_SI,
        'g_dagger_kpc': round(g_dagger_kpc, 6),
        'G_kpc': G_kpc,
        'n_perm': N_PERM,
        'n_boot': N_BOOT,
        'n_shuffle': N_SHUF,
        'detrending': 'UnivariateSpline, s=n*var*0.5',
    },
    'sample': {
        'n_galaxies_total': n_galaxies,
        'n_valid_Lc': n_valid_Lc,
        'n_valid_lambda_short': len(ls_valid),
        'n_rejected': n_rejected,
        'n_missing_mass': n_missing_mass,
    },
    'correlations': correlations,
    'regression_simple': regression_simple,
    'regression_multiple': regression_multi,
    'bootstrap': {
        'slope_ci_95': [round(float(slope_ci[0]), 4), round(float(slope_ci[1]), 4)],
        'rho_ci_95': [round(float(rho_ci[0]), 4), round(float(rho_ci[1]), 4)],
    },
    'permutation_null': {
        'Lc_xi_observed_rho': round(float(obs_rho_Lc_xi), 4) if np.isfinite(obs_rho_Lc_xi) else None,
        'Lc_xi_perm_p': float(perm_p_Lc) if np.isfinite(perm_p_Lc) else None,
        'lambda_xi_observed_rho': round(float(obs_rho_lam_xi), 4) if np.isfinite(obs_rho_lam_xi) else None,
        'lambda_xi_perm_p': float(perm_p_lam) if np.isfinite(perm_p_lam) else None,
    },
    'shuffle_null': {
        'observed_rho': round(float(obs_rho_Lc_xi), 4) if np.isfinite(obs_rho_Lc_xi) else None,
        'shuffled_mean_rho': round(float(shuf_mean), 4) if np.isfinite(shuf_mean) else None,
        'shuffled_std_rho': round(float(shuf_std), 4) if np.isfinite(shuf_std) else None,
        'z_score': round(float(shuf_z), 2) if np.isfinite(shuf_z) else None,
    },
    'verdict': verdict,
    'per_galaxy': per_galaxy_out,
}

outpath = os.path.join(RESULTS_DIR, 'summary_healing_length_scaling.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outpath}")
print("=" * 72)
