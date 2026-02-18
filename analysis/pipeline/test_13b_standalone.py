#!/usr/bin/env python3
"""
Standalone Test 13b: Lensing Profile Shape — Solitonic Core vs NFW

Tests whether Brouwer+2021 KiDS-1000 isolated galaxy ESD profiles
prefer a BEC soliton + NFW envelope model over pure NFW.

Uses precomputed lookup tables for the soliton projected profile
to avoid nested numerical integration inside the optimizer.
"""
import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

print("=" * 72)
print("TEST 13b: LENSING PROFILE SHAPE — SOLITONIC CORE vs NFW")
print("=" * 72)
print("  Direct structural test: condensate ground state via lensing")
print("  Data: Brouwer+2021 KiDS-1000 isolated galaxy ESD profiles")

brouwer_dir = os.path.join(DATA_DIR, 'brouwer2021')

# Stellar mass bin edges: log10(M*/(h70^-2 Msun))
mass_bin_edges = [8.5, 10.3, 10.6, 10.8, 11.0]
mass_bin_labels = [f"[{mass_bin_edges[i]:.1f}, {mass_bin_edges[i+1]:.1f}]"
                   for i in range(4)]
mass_bin_logMs = [0.5 * (mass_bin_edges[i] + mass_bin_edges[i + 1])
                  for i in range(4)]

rc_files = [os.path.join(brouwer_dir,
            f'Fig-3_Lensing-rotation-curves_Massbin-{i+1}.txt')
            for i in range(4)]

all_exist = all(os.path.exists(f) for f in rc_files)
if not all_exist:
    print("ERROR: Brouwer+2021 data files not found!")
    print(f"  Expected in: {brouwer_dir}")
    sys.exit(1)

# Physical constants
G_SI = 6.674e-11
Msun_kg = 1.989e30
kpc_m = 3.086e19
gdagger = 1.2e-10  # m/s^2

# Load all 4 mass bins
esd_data = []
for i, fpath in enumerate(rc_files):
    R_Mpc, ESD_t, ESD_err, bias_K = [], [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            R_Mpc.append(float(parts[0]))
            ESD_t.append(float(parts[1]))
            ESD_err.append(float(parts[3]))
            bias_K.append(float(parts[4]))

    R_Mpc = np.array(R_Mpc)
    ESD_t = np.array(ESD_t)
    ESD_err = np.array(ESD_err)
    bias_K = np.array(bias_K)

    ESD_corr = ESD_t / bias_K
    err_corr = ESD_err / bias_K
    R_kpc = R_Mpc * 1000.0

    esd_data.append({
        'R_Mpc': R_Mpc, 'R_kpc': R_kpc,
        'ESD': ESD_corr, 'err': err_corr,
        'logMs': mass_bin_logMs[i],
        'label': mass_bin_labels[i]
    })

print(f"\n  Loaded {len(esd_data)} stellar mass bins")
for d in esd_data:
    print(f"    logM* {d['label']}: {len(d['R_kpc'])} radial bins, "
          f"R = [{d['R_kpc'][0]:.0f}, {d['R_kpc'][-1]:.0f}] kpc")


# ============================================================
# PRECOMPUTE SOLITON PROJECTED PROFILE (dimensionless)
# ============================================================
# Soliton: ρ(r) = ρ_c [1 + 0.091 (r/r_c)²]⁻⁸
# In dimensionless units u = R/r_c:
#   Σ(u) = ρ_c r_c × σ̃(u)
# where σ̃(u) = 2 ∫₀^∞ [1 + 0.091(u² + t²)]⁻⁸ dt
# and Σ̄(<u) = (2/u²) ∫₀^u u' σ̃(u') du'
# ΔΣ̃(u) = Σ̄(<u) - Σ(u)

print("\n  Precomputing soliton projected profile lookup table...")

N_GRID = 500
u_grid = np.logspace(-3, 3, N_GRID)  # R/r_c from 0.001 to 1000

# Compute σ̃(u) for each grid point
sigma_tilde = np.zeros(N_GRID)
for j, u in enumerate(u_grid):
    def integrand(t):
        return 2.0 * (1.0 + 0.091 * (u**2 + t**2))**(-8)
    val, _ = quad(integrand, 0, 500.0 / max(u, 0.01),
                  limit=200, epsrel=1e-8)
    sigma_tilde[j] = val

# Compute Σ̄(<u) via cumulative integration: Σ̄(<u) = (2/u²) ∫₀^u u' σ̃(u') du'
# Use trapezoidal rule on the fine grid
sigma_bar_tilde = np.zeros(N_GRID)
# Build cumulative integral of u' σ̃(u')
u_sigma_product = u_grid * sigma_tilde
cumul = np.zeros(N_GRID)
for j in range(1, N_GRID):
    du = u_grid[j] - u_grid[j-1]
    cumul[j] = cumul[j-1] + 0.5 * (u_sigma_product[j-1] + u_sigma_product[j]) * du
sigma_bar_tilde = np.where(u_grid > 1e-10, 2.0 * cumul / u_grid**2, sigma_tilde[0])

# ΔΣ̃(u) = Σ̄(<u) - Σ(u)
delta_sigma_tilde = sigma_bar_tilde - sigma_tilde

# Build interpolators (log-space for smoothness)
# Handle potential zeros/negatives carefully
_interp_sigma = interp1d(np.log10(u_grid), sigma_tilde,
                         kind='cubic', fill_value='extrapolate')
_interp_dsigma = interp1d(np.log10(u_grid), delta_sigma_tilde,
                          kind='cubic', fill_value='extrapolate')

print(f"  Lookup table built: {N_GRID} points, u = [{u_grid[0]:.4f}, {u_grid[-1]:.0f}]")
print(f"  Peak Σ̃ = {sigma_tilde.max():.4f} at u = {u_grid[np.argmax(sigma_tilde)]:.4f}")
print(f"  Peak ΔΣ̃ = {delta_sigma_tilde.max():.4f}")


# ============================================================
# MODEL 1: NFW ΔΣ(R) — Wright & Brainerd 2000
# ============================================================
def nfw_delta_sigma(R_kpc, M200_logMsun, c200):
    M200 = 10.0**M200_logMsun
    rho_crit = 1.36e11 * 0.7**2
    r200_Mpc = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
    r_s_Mpc = r200_Mpc / c200
    r_s_kpc = r_s_Mpc * 1000.0
    delta_c = (200.0 / 3.0) * c200**3 / (np.log(1.0 + c200) - c200 / (1.0 + c200))
    rho_s = delta_c * rho_crit
    Sigma_s = rho_s * r_s_Mpc

    x = R_kpc / r_s_kpc
    x = np.clip(x, 1e-6, 1e6)

    Sigma = np.zeros_like(x, dtype=float)
    m1 = x < 1.0
    if np.any(m1):
        t = np.sqrt(1.0 - x[m1]**2)
        Sigma[m1] = 2.0 * Sigma_s / (x[m1]**2 - 1.0) * (
            1.0 / t * np.arccosh(1.0 / x[m1]) - 1.0)
    m2 = np.abs(x - 1.0) < 1e-6
    if np.any(m2):
        Sigma[m2] = 2.0 * Sigma_s / 3.0
    m3 = x > 1.0
    if np.any(m3):
        t = np.sqrt(x[m3]**2 - 1.0)
        Sigma[m3] = 2.0 * Sigma_s / (x[m3]**2 - 1.0) * (
            1.0 - 1.0 / t * np.arctan(t))

    Sigma_bar = np.zeros_like(x, dtype=float)
    m1 = x < 1.0
    if np.any(m1):
        t = np.sqrt(1.0 - x[m1]**2)
        g_x = np.log(x[m1] / 2.0) + 1.0 / t * np.arccosh(1.0 / x[m1])
        Sigma_bar[m1] = 4.0 * Sigma_s * g_x / x[m1]**2
    m2 = np.abs(x - 1.0) < 1e-6
    if np.any(m2):
        Sigma_bar[m2] = 4.0 * Sigma_s * (1.0 + np.log(0.5))
    m3 = x > 1.0
    if np.any(m3):
        t = np.sqrt(x[m3]**2 - 1.0)
        g_x = np.log(x[m3] / 2.0) + 1.0 / t * np.arctan(t)
        Sigma_bar[m3] = 4.0 * Sigma_s * g_x / x[m3]**2

    delta_sigma = (Sigma_bar - Sigma) * 1e-12
    return delta_sigma


# ============================================================
# MODEL 2: BEC Soliton + NFW envelope ΔΣ(R) — FAST VERSION
# ============================================================
def bec_delta_sigma(R_kpc, M200_logMsun, c200, r_c_kpc, f_sol):
    """
    BEC soliton + NFW envelope projected ΔΣ(R).
    Uses precomputed lookup table — no numerical integration in the loop.
    """
    M200 = 10.0**M200_logMsun

    # Soliton mass and central density
    M_sol = f_sol * M200
    rho_c = M_sol / (4.0 * np.pi * 3.883 * r_c_kpc**3)  # Msun/kpc^3

    # Dimensionless radius u = R/r_c
    u = R_kpc / r_c_kpc
    log_u = np.log10(np.clip(u, u_grid[0], u_grid[-1]))

    # ΔΣ_sol = ρ_c × r_c × ΔΣ̃(u), in Msun/kpc^2
    dsigma_sol_kpc2 = rho_c * r_c_kpc * _interp_dsigma(log_u)

    # Convert to Msun/pc^2
    dsigma_sol = dsigma_sol_kpc2 * 1e-6

    # NFW envelope (remaining mass)
    M_env = M200 * (1.0 - f_sol)
    logM_env = np.log10(max(M_env, 1e6))
    dsigma_nfw = nfw_delta_sigma(R_kpc, logM_env, c200)

    return dsigma_sol + dsigma_nfw


# ============================================================
# FIT BOTH MODELS
# ============================================================
print(f"\n  Fitting NFW and BEC soliton+envelope to each mass bin...")

fit_results = []
for i, d in enumerate(esd_data):
    R = d['R_kpc']
    ESD = d['ESD']
    err = d['err']

    valid = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
    R_fit = R[valid]
    ESD_fit = ESD[valid]
    err_fit = err[valid]

    if len(R_fit) < 5:
        print(f"  Bin {i+1} ({d['label']}): insufficient valid points")
        fit_results.append(None)
        continue

    print(f"\n  --- Mass bin {i+1}: logM* = {d['label']} ---")
    print(f"    {len(R_fit)} valid radial points")

    # NFW fit: 2 params
    logMs_i = d['logMs']
    logMh_guess = logMs_i + 1.5
    c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

    def nfw_chi2(params, R_f=R_fit, E_f=ESD_fit, e_f=err_fit):
        logM, c = params
        if c < 1 or c > 50 or logM < 10 or logM > 15:
            return 1e20
        model = nfw_delta_sigma(R_f, logM, c)
        return np.sum(((E_f - model) / e_f)**2)

    res_nfw = minimize(nfw_chi2, [logMh_guess, c_guess],
                       method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-4})
    logM_nfw, c_nfw = res_nfw.x
    chi2_nfw = res_nfw.fun
    aic_nfw = chi2_nfw + 2 * 2

    print(f"    NFW:  logM200={logM_nfw:.2f}, c={c_nfw:.1f}, "
          f"χ²/dof={chi2_nfw / max(len(R_fit) - 2, 1):.3f}, AIC={aic_nfw:.1f}")

    # BEC fit: 4 params
    Ms_SI = 10.0**logMs_i * Msun_kg
    xi_m = np.sqrt(G_SI * Ms_SI / gdagger)
    xi_kpc_pred = xi_m / kpc_m

    # Physical bounds on soliton core radius:
    # The healing length ξ = sqrt(G M* / g†) sets the scale.
    # Allow r_c to range from 0.2× to 5× the predicted ξ.
    # This prevents the optimizer from using r_c as a free parameter
    # to fit noise — it must be physically motivated.
    rc_min = max(0.5, 0.2 * xi_kpc_pred)
    rc_max = max(5.0, 5.0 * xi_kpc_pred)
    print(f"    Predicted ξ = {xi_kpc_pred:.1f} kpc → r_c bounds: "
          f"[{rc_min:.1f}, {rc_max:.1f}] kpc")

    def bec_chi2(params, R_f=R_fit, E_f=ESD_fit, e_f=err_fit):
        logM, c, rc, fs = params
        if (c < 1 or c > 50 or logM < 10 or logM > 15
                or rc < rc_min or rc > rc_max
                or fs < 0.001 or fs > 0.3):
            return 1e20
        try:
            model = bec_delta_sigma(R_f, logM, c, rc, fs)
            if np.any(~np.isfinite(model)):
                return 1e20
            return np.sum(((E_f - model) / e_f)**2)
        except Exception:
            return 1e20

    print(f"    Fitting BEC model (multi-start)...")
    best_bec = None
    best_chi2_bec = 1e20
    # Start at fractions of predicted ξ
    rc_starts = [0.5 * xi_kpc_pred, xi_kpc_pred, 2.0 * xi_kpc_pred,
                 3.0 * xi_kpc_pred]
    fs_starts = [0.01, 0.03, 0.08, 0.15]

    for rc0 in rc_starts:
        for fs0 in fs_starts:
            try:
                res_bec = minimize(
                    bec_chi2,
                    [logMh_guess, c_guess, rc0, fs0],
                    method='Nelder-Mead',
                    options={'maxiter': 30000, 'xatol': 1e-4, 'fatol': 1e-6})
                if res_bec.fun < best_chi2_bec:
                    best_chi2_bec = res_bec.fun
                    best_bec = res_bec.x
            except Exception:
                pass

    if best_bec is not None:
        logM_bec, c_bec, rc_bec, fs_bec = best_bec
        chi2_bec = best_chi2_bec
        aic_bec = chi2_bec + 2 * 4
        n_dof = len(R_fit)

        daic = aic_nfw - aic_bec

        result = {
            'bin': i + 1,
            'label': d['label'],
            'logMs': logMs_i,
            'n_pts': len(R_fit),
            'nfw': {'logM200': logM_nfw, 'c200': c_nfw,
                    'chi2': chi2_nfw, 'aic': aic_nfw,
                    'chi2_dof': chi2_nfw / max(n_dof - 2, 1)},
            'bec': {'logM200': logM_bec, 'c200': c_bec,
                    'r_c_kpc': rc_bec, 'f_sol': fs_bec,
                    'chi2': chi2_bec, 'aic': aic_bec,
                    'chi2_dof': chi2_bec / max(n_dof - 4, 1)},
            'delta_aic': daic,
            'xi_pred_kpc': xi_kpc_pred,
            'bec_preferred': daic > 0
        }
        fit_results.append(result)

        print(f"    BEC:  logM200={logM_bec:.2f}, c={c_bec:.1f}, "
              f"r_c={rc_bec:.1f} kpc, f_sol={fs_bec:.3f}, "
              f"χ²/dof={chi2_bec / max(n_dof - 4, 1):.3f}, AIC={aic_bec:.1f}")
        print(f"    ΔAIC (NFW − BEC) = {daic:+.2f}  "
              f"{'→ BEC preferred' if daic > 0 else '→ NFW preferred'}")
        print(f"    Predicted ξ = {xi_kpc_pred:.1f} kpc, "
              f"fitted r_c = {rc_bec:.1f} kpc "
              f"(ratio: {rc_bec / max(xi_kpc_pred, 0.01):.2f})")
    else:
        print(f"    BEC fit failed")
        fit_results.append(None)

# ============================================================
# AGGREGATE
# ============================================================
valid_results = [r for r in fit_results if r is not None]
n_valid_bins = len(valid_results)

print(f"\n{'=' * 72}")
print(f"AGGREGATE RESULTS")
print(f"{'=' * 72}")

if n_valid_bins >= 2:
    total_daic = sum(r['delta_aic'] for r in valid_results)
    n_bec_preferred = sum(1 for r in valid_results if r['bec_preferred'])

    print(f"  Valid mass bins: {n_valid_bins}/4")
    print(f"  Total ΔAIC (NFW − BEC) = {total_daic:+.2f}")
    print(f"  BEC preferred in {n_bec_preferred}/{n_valid_bins} bins")

    fitted_rc = np.array([r['bec']['r_c_kpc'] for r in valid_results])
    fitted_logMs = np.array([r['logMs'] for r in valid_results])
    predicted_xi = np.array([r['xi_pred_kpc'] for r in valid_results])

    print(f"\n  Core radius scaling:")
    for r in valid_results:
        print(f"    logM*={r['logMs']:.1f}: r_c={r['bec']['r_c_kpc']:.1f} kpc, "
              f"ξ_pred={r['xi_pred_kpc']:.1f} kpc, "
              f"f_sol={r['bec']['f_sol']:.3f}")

    if n_valid_bins >= 3:
        log_rc = np.log10(np.maximum(fitted_rc, 0.1))
        slope, intercept = np.polyfit(fitted_logMs, log_rc, 1)
        if np.std(fitted_rc) > 0 and np.std(predicted_xi) > 0:
            corr, p_corr = pearsonr(fitted_rc, predicted_xi)
        else:
            corr, p_corr = 0.0, 1.0
        print(f"\n  Slope d(log r_c)/d(log M*) = {slope:.3f} (BEC prediction: 0.333)")
        print(f"  Correlation r_c vs ξ_pred: r = {corr:.3f}, p = {p_corr:.4f}")

    # Per-bin detail table
    print(f"\n  {'Bin':>3} {'logM*':>8} {'χ²/dof NFW':>12} {'χ²/dof BEC':>12} "
          f"{'ΔAIC':>8} {'r_c(kpc)':>9} {'f_sol':>7} {'Winner':>8}")
    print(f"  {'-'*73}")
    for r in valid_results:
        winner = "BEC" if r['bec_preferred'] else "NFW"
        print(f"  {r['bin']:>3} {r['logMs']:>8.1f} "
              f"{r['nfw']['chi2_dof']:>12.3f} {r['bec']['chi2_dof']:>12.3f} "
              f"{r['delta_aic']:>+8.2f} {r['bec']['r_c_kpc']:>9.1f} "
              f"{r['bec']['f_sol']:>7.3f} {winner:>8}")

    # Verdict
    if total_daic > 6 and n_bec_preferred >= n_valid_bins // 2 + 1:
        print(f"\n  >>> SOLITONIC CORE DECISIVELY PREFERRED (ΔAIC = {total_daic:+.1f})")
    elif total_daic > 2 and n_bec_preferred >= 2:
        print(f"\n  >>> SOLITONIC CORE PREFERRED (ΔAIC = {total_daic:+.1f})")
    elif total_daic > -2:
        print(f"\n  >>> MODELS INDISTINGUISHABLE (ΔAIC = {total_daic:+.1f})")
    else:
        print(f"\n  >>> NFW PREFERRED (ΔAIC = {total_daic:+.1f})")

    n_physical = sum(1 for r in valid_results
                     if 0.5 < r['bec']['r_c_kpc'] < 100
                     and 0.001 < r['bec']['f_sol'] < 0.3)
    print(f"  Physically reasonable soliton fits: {n_physical}/{n_valid_bins}")
else:
    print(f"  Insufficient valid mass bins: {n_valid_bins}")

print(f"\n{'=' * 72}")
print("Done.")
