#!/usr/bin/env python3
"""
test_gpp_universality.py — GPP Universality Re-test
====================================================

Three-phase test of whether c_s² = g_int·n̄/m is a universal constant:

  Phase A: SPARC high-quality stabilized inversion (spline-smoothed derivatives)
  Phase B: Forward-model comparison (Model U: universal c_s² vs Model G: per-galaxy)
  Phase C: TNG100-1 null test (same inversion on NFW halos — should NOT be universal)

Key question: Is the apparent non-universality from the prior run driven by
noise amplification in the double-derivative, or is it real physics?

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

import json
import os
import sys
import warnings
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import linregress, norm, mannwhitneyu, levene
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(1234)

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')
RAR_FILE = os.path.join(DATA_DIR, 'rar_points_unified.csv')
SPARC_META = os.path.join(DATA_DIR, 'galaxy_results_sparc_orig_haubner.csv')
TNG_FILE = os.path.join(PROJECT_ROOT, 'tng_mass_profiles.npz')

FIG_DIR = os.path.join(RESULTS_DIR, 'gpp_universality')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Constants (SI)
# ============================================================
g_dagger = 1.20e-10          # m/s²
LOG_G_DAGGER = np.log10(g_dagger)

G_SI   = 6.674e-11           # m³ kg⁻¹ s⁻²
M_SUN  = 1.989e30            # kg
KPC_M  = 3.086e19            # m per kpc
HBAR   = 1.0546e-34          # J·s
EV_KG  = 1.783e-36           # kg per eV/c²


# ============================================================
# Data loading
# ============================================================
def load_csv(path):
    """Load CSV into dict of arrays (no pandas dependency)."""
    with open(path) as f:
        header = f.readline().strip().split(',')
        cols = {h: [] for h in header}
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < len(header):
                continue
            for h, v in zip(header, parts):
                cols[h].append(v)
    return cols, header


def safe_float_array(values, default=np.nan):
    """Convert string list to float array, replacing empty/bad values with default."""
    out = np.empty(len(values))
    for i, v in enumerate(values):
        try:
            out[i] = float(v) if v.strip() else default
        except (ValueError, AttributeError):
            out[i] = default
    return out


def load_rar_points():
    cols, _ = load_csv(RAR_FILE)
    n = len(cols['galaxy'])
    out = {
        'galaxy': np.array(cols['galaxy']),
        'source': np.array(cols['source']),
        'log_gbar': safe_float_array(cols['log_gbar']),
        'log_gobs': safe_float_array(cols['log_gobs']),
        'sigma_log_gobs': safe_float_array(cols['sigma_log_gobs'], default=0.1),
        'R_kpc': safe_float_array(cols['R_kpc']),
        'env_dense': np.array(cols['env_dense']),
        'logMh': safe_float_array(cols['logMh'], default=11.0),
    }
    # Drop rows with NaN in critical columns
    valid = (np.isfinite(out['log_gbar']) & np.isfinite(out['log_gobs'])
             & np.isfinite(out['R_kpc']) & (out['R_kpc'] > 0))
    for k in out:
        out[k] = out[k][valid]
    return out


def load_sparc_meta():
    cols, _ = load_csv(SPARC_META)
    meta = {}
    for i in range(len(cols['galaxy'])):
        meta[cols['galaxy'][i]] = {
            'Q': int(cols['Q'][i]),
            'Inc': float(cols['Inc'][i]),
            'fD': int(cols['fD'][i]) if cols['fD'][i] else 0,
            'logMh': float(cols['logMh'][i]),
            'env_binary': cols['env_binary'][i],
            'Vflat': float(cols['Vflat'][i]),
        }
    return meta


def load_tng():
    d = np.load(TNG_FILE, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


# ============================================================
# Core GPP functions
# ============================================================
def spline_derivative(r, y, s_factor=None):
    """Fit smoothing spline to y(r), return (y_smooth, dy_dr) at data points.

    Uses GCV-like smoothing: s = N * var(residuals) if s_factor is None,
    else s = s_factor * N.
    """
    isort = np.argsort(r)
    r_s = r[isort].astype(float)
    y_s = y[isort].astype(float)

    # Remove duplicates (spline requires unique x)
    unique_mask = np.diff(r_s, prepend=-1) > 0
    r_u = r_s[unique_mask]
    y_u = y_s[unique_mask]

    if len(r_u) < 4:
        return np.full_like(r, np.nan), np.full_like(r, np.nan)

    # Smoothing parameter
    if s_factor is None:
        # GCV: start with std-based estimate
        s = len(r_u) * np.var(y_u) * 0.1  # 10% of total variance
    else:
        s = s_factor * len(r_u)

    try:
        spl = UnivariateSpline(r_u, y_u, s=s, k=3)
        y_fit = spl(r_s)
        dy_dr = spl.derivative()(r_s)
    except Exception:
        return np.full_like(r, np.nan), np.full_like(r, np.nan)

    # Map back to original order
    y_out = np.empty_like(r)
    dy_out = np.empty_like(r)
    y_out[isort] = y_fit
    dy_out[isort] = dy_dr
    return y_out, dy_out


def gpp_inversion_galaxy(r_kpc, g_dm, method='spline'):
    """
    Full GPP inversion for a single galaxy.

    Given g_DM(r), computes:
      - M_DM(<r) = g_DM * r² / G
      - ρ_DM(r) = (1/4πGr²) * d(r²·g_DM)/dr   [Poisson]
      - c_s²(r) = |g_DM| / |d ln ρ / dr|         [hydrostatic equil]
      - ξ_eff(r) = r / sqrt(r * |g_DM| / c_s²)   [healing length proxy]

    Returns dict with arrays, or None if too few valid points.
    """
    r_m = r_kpc * KPC_M

    # Sort by radius
    isort = np.argsort(r_m)
    r_s = r_m[isort]
    g_s = g_dm[isort]

    # Only positive g_DM
    pos = g_s > 0
    if pos.sum() < 4:
        return None

    r_pos = r_s[pos]
    g_pos = g_s[pos]

    # M_DM enclosed
    M_enc = g_pos * r_pos**2 / G_SI  # kg

    # Step 2: ρ_DM via spline derivative of r²·g_DM
    r2g = r_pos**2 * g_pos

    if method == 'spline' and len(r_pos) >= 5:
        r2g_smooth, dr2g_dr = spline_derivative(r_pos, r2g)
    else:
        # Savitzky-Golay fallback
        n = len(r_pos)
        win = min(5, n if n % 2 == 1 else n - 1)
        if win >= 3:
            r2g_smooth = savgol_filter(r2g, win, min(2, win - 1))
        else:
            r2g_smooth = r2g
        dr2g_dr = np.gradient(r2g_smooth, r_pos)

    rho = dr2g_dr / (4 * np.pi * G_SI * r_pos**2)

    # Step 3: c_s² via spline derivative of ln(ρ)
    rho_pos = rho > 0
    if rho_pos.sum() < 4:
        return None

    r_rho = r_pos[rho_pos]
    rho_v = rho[rho_pos]
    g_rho = g_pos[rho_pos]
    M_rho = M_enc[rho_pos]
    ln_rho = np.log(rho_v)

    if method == 'spline' and len(r_rho) >= 5:
        _, dln_rho_dr = spline_derivative(r_rho, ln_rho)
    else:
        if len(r_rho) >= 5:
            win = min(5, len(r_rho) if len(r_rho) % 2 == 1 else len(r_rho) - 1)
            if win >= 3:
                ln_rho_s = savgol_filter(ln_rho, win, min(2, win - 1))
            else:
                ln_rho_s = ln_rho
        else:
            ln_rho_s = ln_rho
        dln_rho_dr = np.gradient(ln_rho_s, r_rho)

    abs_dln = np.abs(dln_rho_dr)
    valid = (abs_dln > 1e-35) & np.isfinite(dln_rho_dr) & (g_rho > 0)

    if valid.sum() < 2:
        return None

    cs2 = np.abs(g_rho[valid]) / abs_dln[valid]  # m²/s²
    r_out = r_rho[valid] / KPC_M  # back to kpc
    M_out = M_rho[valid]

    # ξ_eff = ℏ / (m * c_s * sqrt(2))  but we don't know m.
    # Use dimensional proxy: ξ_proxy = c_s² / |g_DM|  [length]
    xi_proxy = cs2 / np.abs(g_rho[valid])  # meters

    # X = r / ξ_proxy
    X = (r_rho[valid]) / xi_proxy

    return {
        'r_kpc': r_out,
        'cs2': cs2,
        'M_enc_kg': M_out,
        'rho': rho_v[valid] if valid.sum() <= len(rho_v) else rho_v[:valid.sum()],
        'xi_proxy_kpc': xi_proxy / KPC_M,
        'X': X,
        'n_valid': int(valid.sum()),
        'n_pos_gdm': int(pos.sum()),
        'neg_gdm_frac': 1 - pos.sum() / len(g_s),
    }


# ============================================================
# PHASE A: SPARC High-Quality Stabilized Inversion
# ============================================================
def phase_a(pts, sparc_meta):
    print("\n" + "=" * 70)
    print("PHASE A: SPARC HIGH-QUALITY STABILIZED INVERSION")
    print("=" * 70)

    # Group SPARC points by galaxy
    sparc_mask = pts['source'] == 'SPARC'
    sparc_idx = np.where(sparc_mask)[0]

    gal_groups = {}
    for i in sparc_idx:
        g = pts['galaxy'][i]
        if g not in gal_groups:
            gal_groups[g] = []
        gal_groups[g].append(i)

    # Apply quality cuts
    hq_galaxies = []
    for gname, idx_list in gal_groups.items():
        n_raw = len(idx_list)
        if n_raw < 10:
            continue
        if gname not in sparc_meta:
            continue
        m = sparc_meta[gname]
        if m['Q'] > 1:
            continue
        if not (45 <= m['Inc'] <= 75):
            continue
        if m['fD'] not in (1, 2):
            continue
        hq_galaxies.append((gname, np.array(idx_list), m))

    print(f"  SPARC galaxies total: {len(gal_groups)}")
    print(f"  After HQ cuts (Q=1, 45≤Inc≤75, fD∈{{1,2}}, N≥10): {len(hq_galaxies)}")

    # Also prepare a relaxed sample (Q≤2, 30<Inc<85) for comparison
    relaxed_galaxies = []
    for gname, idx_list in gal_groups.items():
        n_raw = len(idx_list)
        if n_raw < 8:
            continue
        if gname not in sparc_meta:
            continue
        m = sparc_meta[gname]
        if m['Q'] > 2:
            continue
        if not (30 < m['Inc'] < 85):
            continue
        relaxed_galaxies.append((gname, np.array(idx_list), m))

    print(f"  Relaxed sample (Q≤2, 30<Inc<85, N≥8): {len(relaxed_galaxies)}")

    # Run inversion on both samples
    results = {}
    for label, sample in [('HQ', hq_galaxies), ('relaxed', relaxed_galaxies)]:
        gal_names = []
        gal_cs2 = []
        gal_cs2_mad = []
        gal_logMh = []
        gal_env = []
        gal_npts = []
        gal_tf_frac = []
        gal_profiles = {}

        for gname, idx, meta in sample:
            r_kpc = pts['R_kpc'][idx]
            gbar = 10.0**pts['log_gbar'][idx]
            gobs = 10.0**pts['log_gobs'][idx]
            g_dm = gobs - gbar

            inv = gpp_inversion_galaxy(r_kpc, g_dm, method='spline')
            if inv is None:
                continue

            cs2_med = np.median(inv['cs2'])
            log_cs2 = np.log10(inv['cs2'])
            cs2_mad = 1.4826 * np.median(np.abs(log_cs2 - np.median(log_cs2)))

            tf_frac = (inv['X'] > 1).sum() / len(inv['X']) if len(inv['X']) > 0 else 0

            gal_names.append(gname)
            gal_cs2.append(cs2_med)
            gal_cs2_mad.append(cs2_mad)
            gal_logMh.append(meta['logMh'])
            gal_env.append(meta['env_binary'])
            gal_npts.append(inv['n_valid'])
            gal_tf_frac.append(tf_frac)
            gal_profiles[gname] = inv

        gal_cs2 = np.array(gal_cs2)
        gal_logMh = np.array(gal_logMh)
        gal_env = np.array(gal_env)
        gal_npts = np.array(gal_npts)
        gal_tf_frac = np.array(gal_tf_frac)

        # Statistics
        log_cs2 = np.log10(gal_cs2)
        finite = np.isfinite(log_cs2) & np.isfinite(gal_logMh)
        lc = log_cs2[finite]
        lm = gal_logMh[finite]
        env = gal_env[finite]

        mean_lc = np.mean(lc)
        std_lc = np.std(lc)
        med_cs2 = np.median(gal_cs2[finite])

        if len(lc) >= 5:
            sl = linregress(lm, lc)
            slope, slope_se, slope_p = sl.slope, sl.stderr, sl.pvalue
        else:
            slope = slope_se = slope_p = np.nan

        # Environment
        field_mask = env == 'field'
        dense_mask = env == 'dense'
        env_result = {}
        if field_mask.sum() >= 5 and dense_mask.sum() >= 5:
            try:
                _, mw_p = mannwhitneyu(lc[field_mask], lc[dense_mask], alternative='two-sided')
                _, lev_p = levene(lc[field_mask], lc[dense_mask])
            except Exception:
                mw_p = lev_p = np.nan
            env_result = {
                'field_n': int(field_mask.sum()),
                'dense_n': int(dense_mask.sum()),
                'field_mean': round(float(np.mean(lc[field_mask])), 4),
                'dense_mean': round(float(np.mean(lc[dense_mask])), 4),
                'field_std': round(float(np.std(lc[field_mask])), 4),
                'dense_std': round(float(np.std(lc[dense_mask])), 4),
                'mann_whitney_p': round(float(mw_p), 4) if np.isfinite(mw_p) else None,
                'levene_p': round(float(lev_p), 4) if np.isfinite(lev_p) else None,
            }
        else:
            env_result = {
                'field_n': int(field_mask.sum()),
                'dense_n': int(dense_mask.sum()),
                'note': 'Too few dense galaxies for env test'
            }

        results[label] = {
            'n_galaxies': int(finite.sum()),
            'n_points_total': int(gal_npts.sum()),
            'cs2_median': float(f"{med_cs2:.6e}"),
            'cs2_mean_log10': round(float(mean_lc), 4),
            'cs2_std_dex': round(float(std_lc), 4),
            'mass_slope': round(float(slope), 5),
            'mass_slope_se': round(float(slope_se), 5),
            'mass_slope_p': float(f"{slope_p:.4e}") if np.isfinite(slope_p) else None,
            'mean_TF_fraction': round(float(np.mean(gal_tf_frac)), 4),
            'env_split': env_result,
            'profiles': gal_profiles,
            'log_cs2': lc,
            'logMh': lm,
            'env': env,
            'gal_names': np.array(gal_names)[finite],
        }

        print(f"\n  [{label.upper()}] {finite.sum()} galaxies, {gal_npts.sum()} points")
        print(f"    Median c_s²: {med_cs2:.4e} m²/s²")
        print(f"    log₁₀(c_s²): {mean_lc:.3f} ± {std_lc:.3f} dex")
        print(f"    Mass slope: {slope:.4f} ± {slope_se:.4f} (p={slope_p:.3e})")
        print(f"    Mean TF fraction: {np.mean(gal_tf_frac):.2f}")
        print(f"    Env: {env_result}")

    return results


# ============================================================
# PHASE B: Forward-Model Comparison
# ============================================================
def phase_b(pts, sparc_meta):
    """
    Forward model avoids double-derivative entirely.

    Instead of: data → ρ_DM → d ln ρ/dr → c_s²  (two derivatives)
    We use: data → M_DM(<r) → d ln M_DM/dr → c_s²_proxy  (one derivative)

    c_s²_proxy(r) = g_DM(r) · r / |d ln M_DM(<r)/d ln r|

    Justification: For a polytropic halo in hydrostatic equilibrium,
    d ln M/d ln r = r²·ρ·4πG / g_DM. If P = c_s²·ρ (isothermal-like),
    then the Lane-Emden structure gives c_s² ∝ g_DM · r / (d ln M/d ln r).
    This is exact for an isothermal sphere where d ln M/d ln r → 1 at large r.

    Model U: c_s² is a single universal constant C for all galaxies.
    Model G: c_s² is free per galaxy (= per-galaxy median c_s²_proxy).
    """
    print("\n" + "=" * 70)
    print("PHASE B: FORWARD-MODEL COMPARISON (SINGLE-DERIVATIVE)")
    print("=" * 70)

    # Use the relaxed SPARC sample for more galaxies
    sparc_mask = pts['source'] == 'SPARC'
    sparc_idx = np.where(sparc_mask)[0]

    gal_groups = {}
    for i in sparc_idx:
        g = pts['galaxy'][i]
        if g not in gal_groups:
            gal_groups[g] = []
        gal_groups[g].append(i)

    gal_cs2_proxy = []
    gal_logMh_b = []
    gal_names_b = []
    gal_npts_b = []
    all_log_cs2_proxy = []  # All per-point values for Model U likelihood
    all_sigma = []

    for gname, idx_list in gal_groups.items():
        idx = np.array(idx_list)
        if len(idx) < 8:
            continue
        if gname not in sparc_meta:
            continue
        m = sparc_meta[gname]
        if m['Q'] > 2:
            continue

        r_kpc = pts['R_kpc'][idx]
        gbar = 10.0**pts['log_gbar'][idx]
        gobs = 10.0**pts['log_gobs'][idx]
        sigma = pts['sigma_log_gobs'][idx]
        g_dm = gobs - gbar

        # Sort by radius
        isort = np.argsort(r_kpc)
        r_s = r_kpc[isort]
        g_s = g_dm[isort]
        sig_s = sigma[isort]

        # Only positive g_DM
        pos = g_s > 0
        if pos.sum() < 5:
            continue

        r_pos = r_s[pos]
        g_pos = g_s[pos]
        sig_pos = sig_s[pos]

        r_m = r_pos * KPC_M
        M_enc = g_pos * r_m**2 / G_SI  # kg

        # d ln M / d ln r via spline
        ln_r = np.log(r_m)
        ln_M = np.log(np.maximum(M_enc, 1e-30))

        if len(r_pos) >= 5:
            _, dln_M_dln_r = spline_derivative(ln_r, ln_M)
        else:
            dln_M_dln_r = np.gradient(ln_M, ln_r)

        valid = np.isfinite(dln_M_dln_r) & (np.abs(dln_M_dln_r) > 0.01)
        if valid.sum() < 3:
            continue

        # c_s²_proxy = g_DM · r / |d ln M / d ln r|
        cs2_proxy = np.abs(g_pos[valid]) * r_m[valid] / np.abs(dln_M_dln_r[valid])

        cs2_valid = cs2_proxy[cs2_proxy > 0]
        if len(cs2_valid) < 3:
            continue

        med = np.median(cs2_valid)
        gal_cs2_proxy.append(med)
        gal_logMh_b.append(m['logMh'])
        gal_names_b.append(gname)
        gal_npts_b.append(len(cs2_valid))

        all_log_cs2_proxy.extend(np.log10(cs2_valid).tolist())
        all_sigma.extend(sig_pos[valid][:len(cs2_valid)].tolist())

    gal_cs2_proxy = np.array(gal_cs2_proxy)
    gal_logMh_b = np.array(gal_logMh_b)
    gal_npts_b = np.array(gal_npts_b)
    all_log_cs2 = np.array(all_log_cs2_proxy)

    n_gal = len(gal_cs2_proxy)
    log_cs2_gal = np.log10(gal_cs2_proxy)
    finite = np.isfinite(log_cs2_gal) & np.isfinite(gal_logMh_b)

    print(f"  Galaxies with valid c_s²_proxy: {finite.sum()}")

    lc = log_cs2_gal[finite]
    lm = gal_logMh_b[finite]

    mean_lc = np.mean(lc)
    std_lc = np.std(lc)
    med_cs2 = np.median(gal_cs2_proxy[finite])

    sl = linregress(lm, lc)
    print(f"  Median c_s²_proxy: {med_cs2:.4e} m²/s²")
    print(f"  log₁₀(c_s²_proxy): {mean_lc:.3f} ± {std_lc:.3f} dex")
    print(f"  Mass slope: {sl.slope:.4f} ± {sl.stderr:.4f} (p={sl.pvalue:.3e})")

    # Model comparison: AIC/BIC
    # Model U: all galaxies share ONE c_s², residuals = log_cs2_gal - C_global
    C_global = np.median(lc)
    resid_U = lc - C_global
    sigma_U = np.std(resid_U)
    # Log-likelihood for Model U (1 parameter: C_global)
    LL_U = -0.5 * np.sum((resid_U / sigma_U)**2) - len(lc) * np.log(sigma_U * np.sqrt(2 * np.pi))
    k_U = 2  # C_global + sigma_U

    # Model G: each galaxy has its own c_s², residuals = 0 by construction
    # But use within-galaxy scatter as the per-galaxy residual
    # Per-galaxy: each median is a free parameter, within-galaxy spread is the residual
    # Effective residuals: within-galaxy MAD
    per_gal_scatter = []
    for i in range(n_gal):
        if not finite[i]:
            continue
        n_i = gal_npts_b[i]
        per_gal_scatter.append(n_i)  # Each galaxy has n_i - 1 dof used

    k_G = finite.sum() + 1  # N_gal free parameters + 1 global sigma
    # For Model G, the between-galaxy variance is zero (it's fit exactly)
    # The penalty is in the number of parameters
    sigma_G = 0.01  # Effectively zero between-galaxy residual
    # But the actual residual is within-galaxy, which Model U also has
    # Fair comparison: both models have within-galaxy scatter as irreducible noise.
    # The MODEL DIFFERENCE is in the between-galaxy variance.

    # Better: use AIC based on galaxy-level data
    # Model U: log_cs2_gal ~ N(C, σ²), k=2
    LL_U = np.sum(norm.logpdf(lc, loc=C_global, scale=sigma_U))
    AIC_U = 2 * k_U - 2 * LL_U
    BIC_U = k_U * np.log(len(lc)) - 2 * LL_U

    # Model G: log_cs2_gal = free per galaxy. With N_gal params, LL → perfect fit
    # Use per-galaxy within-galaxy MAD as the "uncertainty"
    # This is a saturated model — AIC penalizes heavily
    sigma_G_arr = np.array([0.001] * len(lc))  # Effectively zero residual
    LL_G = 0  # Perfect fit by construction
    AIC_G = 2 * k_G - 2 * LL_G  # Large penalty
    BIC_G = k_G * np.log(len(lc)) - 2 * LL_G

    # More rigorous: fit linear model as Model G proxy
    # Model L: log_cs2 = a + b*logMh, k=3
    k_L = 3
    pred_L = sl.intercept + sl.slope * lm
    resid_L = lc - pred_L
    sigma_L = np.std(resid_L)
    LL_L = np.sum(norm.logpdf(lc, loc=pred_L, scale=sigma_L))
    AIC_L = 2 * k_L - 2 * LL_L
    BIC_L = k_L * np.log(len(lc)) - 2 * LL_L

    delta_AIC = AIC_U - AIC_L  # positive → Model L (mass-dependent) preferred
    w_U = np.exp(-0.5 * min(0, delta_AIC)) / (np.exp(-0.5 * min(0, delta_AIC)) + np.exp(-0.5 * max(0, -delta_AIC)))

    print(f"\n  Model comparison (galaxy-level):")
    print(f"    Model U (universal): AIC={AIC_U:.1f}, BIC={BIC_U:.1f}")
    print(f"    Model L (linear Mh): AIC={AIC_L:.1f}, BIC={BIC_L:.1f}")
    print(f"    ΔAIC (U - L): {delta_AIC:.1f} ({'U preferred' if delta_AIC < 0 else 'L preferred'})")
    print(f"    Akaike weight for U: {w_U:.4f}")

    return {
        'n_galaxies': int(finite.sum()),
        'n_points': int(gal_npts_b[finite].sum()),
        'cs2_proxy_median': float(f"{med_cs2:.6e}"),
        'cs2_proxy_mean_log10': round(float(mean_lc), 4),
        'cs2_proxy_std_dex': round(float(std_lc), 4),
        'mass_slope': round(float(sl.slope), 5),
        'mass_slope_se': round(float(sl.stderr), 5),
        'mass_slope_p': float(f"{sl.pvalue:.4e}"),
        'AIC_universal': round(float(AIC_U), 2),
        'AIC_linear_mass': round(float(AIC_L), 2),
        'BIC_universal': round(float(BIC_U), 2),
        'BIC_linear_mass': round(float(BIC_L), 2),
        'delta_AIC_U_minus_L': round(float(delta_AIC), 2),
        'akaike_weight_universal': round(float(w_U), 4),
        'log_cs2': lc,
        'logMh': lm,
    }


# ============================================================
# PHASE C: TNG100-1 NULL TEST
# ============================================================
def phase_c(tng_data):
    """
    Apply same GPP inversion to TNG100-1 NFW halos.

    TNG halos are NOT BEC — they are collisionless ΛCDM dark matter.
    If c_s² comes out universal in TNG, the test is non-discriminating.
    If c_s² is mass-dependent / scattered in TNG, that's evidence the
    SPARC result (if universal) is physically meaningful.
    """
    print("\n" + "=" * 70)
    print("PHASE C: TNG100-1 NULL TEST (ΛCDM NFW HALOS)")
    print("=" * 70)

    galaxy_ids = tng_data['galaxy_ids']
    r_half = tng_data['r_half_kpc']
    vmax = tng_data['vmax']
    m_star_total = tng_data['m_star_total']
    radii = tng_data['radii_kpc']       # (N, 15)
    m_star_enc = tng_data['m_star_enc']  # (N, 15)
    m_gas_enc = tng_data['m_gas_enc']    # (N, 15)
    m_dm_enc = tng_data['m_dm_enc']      # (N, 15)

    n_gal = len(galaxy_ids)
    print(f"  TNG galaxies loaded: {n_gal}")

    # Quality cuts: need positive DM mass, enough radial range
    gal_cs2 = []
    gal_cs2_std = []
    gal_logMstar = []
    gal_logMhalo = []  # total mass as halo proxy
    gal_vmax = []
    gal_tf_frac = []
    gal_npts = []
    gal_profiles = {}

    n_skip_dm = 0
    n_skip_inv = 0

    for i in range(n_gal):
        r_kpc = radii[i]
        ms = m_star_enc[i]
        mg = m_gas_enc[i]
        md = m_dm_enc[i]

        # Check for valid DM mass
        if md[-1] <= 0 or ms[-1] <= 0:
            n_skip_dm += 1
            continue

        # Only use radii where DM mass is positive and increasing
        valid = (md > 0) & (r_kpc > 0)
        if valid.sum() < 5:
            n_skip_dm += 1
            continue

        r_v = r_kpc[valid]
        md_v = md[valid]
        ms_v = ms[valid]
        mg_v = mg[valid]

        # g_bar and g_obs in SI
        r_m = r_v * KPC_M
        m_bar_kg = (ms_v + mg_v) * M_SUN
        m_tot_kg = (ms_v + mg_v + md_v) * M_SUN
        m_dm_kg = md_v * M_SUN

        g_bar = G_SI * m_bar_kg / r_m**2
        g_obs = G_SI * m_tot_kg / r_m**2
        g_dm = g_obs - g_bar

        # Run GPP inversion
        inv = gpp_inversion_galaxy(r_v, g_dm, method='spline')
        if inv is None:
            n_skip_inv += 1
            continue

        cs2_med = np.median(inv['cs2'])
        cs2_std = np.std(np.log10(inv['cs2']))
        tf_frac = (inv['X'] > 1).sum() / len(inv['X']) if len(inv['X']) > 0 else 0

        # Halo mass proxy = total mass at outermost radius
        M_halo_proxy = (ms[-1] + mg[-1] + md[-1])

        gal_cs2.append(cs2_med)
        gal_cs2_std.append(cs2_std)
        gal_logMstar.append(np.log10(max(ms[-1], 1)))
        gal_logMhalo.append(np.log10(max(M_halo_proxy, 1)))
        gal_vmax.append(vmax[i])
        gal_tf_frac.append(tf_frac)
        gal_npts.append(inv['n_valid'])

        if len(gal_cs2) <= 20:
            gal_profiles[int(galaxy_ids[i])] = inv

    gal_cs2 = np.array(gal_cs2)
    gal_logMhalo = np.array(gal_logMhalo)
    gal_logMstar = np.array(gal_logMstar)
    gal_vmax = np.array(gal_vmax)
    gal_npts = np.array(gal_npts)
    gal_tf_frac = np.array(gal_tf_frac)

    print(f"  Processed: {len(gal_cs2)} galaxies")
    print(f"  Skipped (no DM): {n_skip_dm}")
    print(f"  Skipped (inversion failed): {n_skip_inv}")

    log_cs2 = np.log10(gal_cs2)
    finite = np.isfinite(log_cs2) & np.isfinite(gal_logMhalo)
    lc = log_cs2[finite]
    lm = gal_logMhalo[finite]
    lms = gal_logMstar[finite]

    mean_lc = np.mean(lc)
    std_lc = np.std(lc)
    med_cs2 = np.median(gal_cs2[finite])

    sl = linregress(lm, lc)
    sl_star = linregress(lms, lc)

    print(f"\n  Median c_s²: {med_cs2:.4e} m²/s²")
    print(f"  log₁₀(c_s²): {mean_lc:.3f} ± {std_lc:.3f} dex")
    print(f"  Mass slope (vs M_halo): {sl.slope:.4f} ± {sl.stderr:.4f} (p={sl.pvalue:.3e})")
    print(f"  Mass slope (vs M_star): {sl_star.slope:.4f} ± {sl_star.stderr:.4f} (p={sl_star.pvalue:.3e})")
    print(f"  Mean TF fraction: {np.mean(gal_tf_frac):.2f}")

    # Model comparison for TNG
    C_global = np.median(lc)
    resid_U = lc - C_global
    sigma_U = np.std(resid_U)
    LL_U = np.sum(norm.logpdf(lc, loc=C_global, scale=sigma_U))
    AIC_U = 2 * 2 - 2 * LL_U
    BIC_U = 2 * np.log(len(lc)) - 2 * LL_U

    pred_L = sl.intercept + sl.slope * lm
    sigma_L = np.std(lc - pred_L)
    LL_L = np.sum(norm.logpdf(lc, loc=pred_L, scale=sigma_L))
    AIC_L = 2 * 3 - 2 * LL_L
    BIC_L = 3 * np.log(len(lc)) - 2 * LL_L

    delta_AIC = AIC_U - AIC_L

    print(f"\n  TNG Model comparison:")
    print(f"    Model U (universal): AIC={AIC_U:.1f}")
    print(f"    Model L (linear Mh): AIC={AIC_L:.1f}")
    print(f"    ΔAIC (U - L): {delta_AIC:.1f} ({'U preferred' if delta_AIC < 0 else 'L preferred'})")

    return {
        'n_galaxies': int(finite.sum()),
        'cs2_median': float(f"{med_cs2:.6e}"),
        'cs2_mean_log10': round(float(mean_lc), 4),
        'cs2_std_dex': round(float(std_lc), 4),
        'mass_slope_halo': round(float(sl.slope), 5),
        'mass_slope_halo_se': round(float(sl.stderr), 5),
        'mass_slope_halo_p': float(f"{sl.pvalue:.4e}"),
        'mass_slope_star': round(float(sl_star.slope), 5),
        'mass_slope_star_p': float(f"{sl_star.pvalue:.4e}"),
        'AIC_universal': round(float(AIC_U), 2),
        'AIC_linear_mass': round(float(AIC_L), 2),
        'delta_AIC': round(float(delta_AIC), 2),
        'mean_TF_fraction': round(float(np.mean(gal_tf_frac)), 4),
        'log_cs2': lc,
        'logMh': lm,
        'logMstar': lms,
        'profiles': gal_profiles,
    }


# ============================================================
# Plotting
# ============================================================
def make_plots(phase_a_res, phase_b_res, phase_c_res):
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.30,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    # ---- Row 1: SPARC HQ c_s² vs logMh, histogram, scatter comparison ----

    # Plot 1a: SPARC HQ c_s² vs logMh
    ax = fig.add_subplot(gs[0, 0])
    hq = phase_a_res['HQ']
    lc, lm = hq['log_cs2'], hq['logMh']
    env = hq['env']
    colors = np.where(env == 'dense', 'red', 'steelblue')
    ax.scatter(lm, lc, c=colors, s=20, alpha=0.6, edgecolors='k', linewidths=0.3)
    sl = linregress(lm, lc)
    x_fit = np.linspace(lm.min(), lm.max(), 50)
    ax.plot(x_fit, sl.intercept + sl.slope * x_fit, 'k-', lw=2,
            label=f'slope={sl.slope:.3f}±{sl.stderr:.3f}\np={sl.pvalue:.2e}')
    ax.axhline(np.mean(lc), color='red', ls='--', lw=1)
    ax.set_xlabel('log₁₀(M_halo) [M☉]')
    ax.set_ylabel('log₁₀(c_s²) [m²/s²]')
    ax.set_title(f'A1: SPARC HQ (N={len(lc)}, σ={np.std(lc):.2f} dex)')
    ax.legend(fontsize=7)

    # Plot 1b: SPARC HQ histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(lc, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='k', lw=0.5)
    x_g = np.linspace(lc.min() - 0.5, lc.max() + 0.5, 200)
    ax.plot(x_g, norm.pdf(x_g, np.mean(lc), np.std(lc)), 'r-', lw=2,
            label=f'μ={np.mean(lc):.2f}\nσ={np.std(lc):.2f} dex')
    ax.set_xlabel('log₁₀(c_s²) [m²/s²]')
    ax.set_ylabel('Density')
    ax.set_title('A2: SPARC HQ c_s² distribution')
    ax.legend(fontsize=8)

    # Plot 1c: Scatter comparison bar chart
    ax = fig.add_subplot(gs[0, 2])
    labels_bar = ['Prior\ninversion\n(all)', 'Phase A\nHQ', 'Phase A\nrelaxed', 'Phase B\nproxy']
    scatter_vals = [
        0.623,  # prior run
        phase_a_res['HQ']['cs2_std_dex'],
        phase_a_res['relaxed']['cs2_std_dex'],
        phase_b_res['cs2_proxy_std_dex'],
    ]
    bar_colors = ['gray', 'steelblue', 'orange', 'green']
    bars = ax.bar(labels_bar, scatter_vals, color=bar_colors, edgecolor='k', lw=0.5)
    ax.axhline(0.2, color='red', ls='--', lw=1, label='BEC target (0.2 dex)')
    ax.set_ylabel('σ(log₁₀ c_s²) [dex]')
    ax.set_title('A3: Scatter reduction')
    ax.legend(fontsize=8)
    for b, v in zip(bars, scatter_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)

    # ---- Row 2: Phase B forward model ----

    ax = fig.add_subplot(gs[1, 0])
    lc_b, lm_b = phase_b_res['log_cs2'], phase_b_res['logMh']
    ax.scatter(lm_b, lc_b, s=10, alpha=0.4, c='green', edgecolors='none')
    sl_b = linregress(lm_b, lc_b)
    x_fit = np.linspace(lm_b.min(), lm_b.max(), 50)
    ax.plot(x_fit, sl_b.intercept + sl_b.slope * x_fit, 'k-', lw=2,
            label=f'slope={sl_b.slope:.3f}±{sl_b.stderr:.3f}')
    ax.axhline(np.mean(lc_b), color='red', ls='--', lw=1)
    ax.set_xlabel('log₁₀(M_halo) [M☉]')
    ax.set_ylabel('log₁₀(c_s²_proxy) [m²/s²]')
    ax.set_title(f'B1: Forward proxy (N={len(lc_b)}, σ={np.std(lc_b):.2f} dex)')
    ax.legend(fontsize=7)

    # Phase B histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(lc_b, bins=20, density=True, alpha=0.7, color='green', edgecolor='k', lw=0.5)
    x_g = np.linspace(lc_b.min() - 0.5, lc_b.max() + 0.5, 200)
    ax.plot(x_g, norm.pdf(x_g, np.mean(lc_b), np.std(lc_b)), 'r-', lw=2,
            label=f'μ={np.mean(lc_b):.2f}\nσ={np.std(lc_b):.2f}')
    ax.set_xlabel('log₁₀(c_s²_proxy) [m²/s²]')
    ax.set_ylabel('Density')
    ax.set_title('B2: Proxy distribution')
    ax.legend(fontsize=8)

    # Phase B AIC comparison
    ax = fig.add_subplot(gs[1, 2])
    aic_vals = [phase_b_res['AIC_universal'], phase_b_res['AIC_linear_mass']]
    aic_labels = ['Model U\n(universal)', 'Model L\n(linear Mh)']
    bars = ax.bar(aic_labels, aic_vals, color=['steelblue', 'orange'], edgecolor='k', lw=0.5)
    ax.set_ylabel('AIC')
    daic = phase_b_res['delta_AIC_U_minus_L']
    ax.set_title(f'B3: ΔAIC(U−L) = {daic:.1f} ({"U" if daic < 0 else "L"} preferred)')
    for b, v in zip(bars, aic_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)

    # ---- Row 3: TNG null test ----

    ax = fig.add_subplot(gs[2, 0])
    lc_t, lm_t = phase_c_res['log_cs2'], phase_c_res['logMh']
    # Subsample for plotting if too many
    if len(lc_t) > 2000:
        rng = np.random.RandomState(42)
        idx_plot = rng.choice(len(lc_t), 2000, replace=False)
    else:
        idx_plot = np.arange(len(lc_t))
    ax.scatter(lm_t[idx_plot], lc_t[idx_plot], s=3, alpha=0.15, c='purple', edgecolors='none')
    sl_t = linregress(lm_t, lc_t)
    x_fit = np.linspace(lm_t.min(), lm_t.max(), 50)
    ax.plot(x_fit, sl_t.intercept + sl_t.slope * x_fit, 'k-', lw=2,
            label=f'slope={sl_t.slope:.3f}±{sl_t.stderr:.3f}')
    ax.axhline(np.mean(lc_t), color='red', ls='--', lw=1)
    ax.set_xlabel('log₁₀(M_halo) [M☉]')
    ax.set_ylabel('log₁₀(c_s²) [m²/s²]')
    ax.set_title(f'C1: TNG ΛCDM (N={len(lc_t)}, σ={np.std(lc_t):.2f} dex)')
    ax.legend(fontsize=7)

    # TNG histogram
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(lc_t, bins=50, density=True, alpha=0.7, color='purple', edgecolor='k', lw=0.3)
    x_g = np.linspace(np.percentile(lc_t, 2), np.percentile(lc_t, 98), 200)
    ax.plot(x_g, norm.pdf(x_g, np.mean(lc_t), np.std(lc_t)), 'r-', lw=2,
            label=f'μ={np.mean(lc_t):.2f}\nσ={np.std(lc_t):.2f}')
    ax.set_xlabel('log₁₀(c_s²) [m²/s²]')
    ax.set_ylabel('Density')
    ax.set_title('C2: TNG c_s² distribution')
    ax.legend(fontsize=8)

    # TNG vs SPARC overlay
    ax = fig.add_subplot(gs[2, 2])
    # Normalized histograms
    bins_range = (min(lc.min(), lc_t.min()) - 0.5, max(lc.max(), lc_t.max()) + 0.5)
    bins_n = 30
    ax.hist(lc, bins=bins_n, range=bins_range, density=True, alpha=0.5,
            color='steelblue', edgecolor='k', lw=0.3, label=f'SPARC HQ (σ={np.std(lc):.2f})')
    ax.hist(lc_t, bins=bins_n, range=bins_range, density=True, alpha=0.3,
            color='purple', edgecolor='k', lw=0.3, label=f'TNG ΛCDM (σ={np.std(lc_t):.2f})')
    ax.set_xlabel('log₁₀(c_s²) [m²/s²]')
    ax.set_ylabel('Density')
    ax.set_title('C3: SPARC vs TNG comparison')
    ax.legend(fontsize=8)

    # ---- Row 4: Discrimination summary ----

    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Build summary table
    sparc_hq = phase_a_res['HQ']
    text_lines = [
        "GPP UNIVERSALITY TEST — SUMMARY",
        "─" * 80,
        f"{'Metric':<35s} {'SPARC HQ':>12s} {'SPARC proxy':>12s} {'TNG ΛCDM':>12s} {'Prior run':>12s}",
        "─" * 80,
        f"{'N galaxies':<35s} {sparc_hq['n_galaxies']:>12d} {phase_b_res['n_galaxies']:>12d} {phase_c_res['n_galaxies']:>12d} {'627':>12s}",
        f"{'σ(log c_s²) [dex]':<35s} {sparc_hq['cs2_std_dex']:>12.3f} {phase_b_res['cs2_proxy_std_dex']:>12.3f} {phase_c_res['cs2_std_dex']:>12.3f} {'0.623':>12s}",
        f"{'Mass slope [dex/dex]':<35s} {sparc_hq['mass_slope']:>12.4f} {phase_b_res['mass_slope']:>12.4f} {phase_c_res['mass_slope_halo']:>12.4f} {'0.0291':>12s}",
        f"{'Mass slope p-value':<35s} {str(sparc_hq['mass_slope_p']):>12s} {str(phase_b_res['mass_slope_p']):>12s} {str(phase_c_res['mass_slope_halo_p']):>12s} {'0.032':>12s}",
        f"{'ΔAIC (U−L)':<35s} {'—':>12s} {phase_b_res['delta_AIC_U_minus_L']:>12.1f} {phase_c_res['delta_AIC']:>12.1f} {'—':>12s}",
        "─" * 80,
    ]

    # Verdict logic
    sparc_scatter = sparc_hq['cs2_std_dex']
    tng_scatter = phase_c_res['cs2_std_dex']
    sparc_slope_p = sparc_hq['mass_slope_p'] if sparc_hq['mass_slope_p'] is not None else 1.0

    if sparc_scatter < tng_scatter * 0.7 and sparc_slope_p > 0.05:
        verdict = "DISCRIMINATING — SPARC more universal than TNG (BEC-consistent)"
    elif sparc_scatter < tng_scatter and sparc_slope_p > 0.01:
        verdict = "MARGINAL — SPARC tighter than TNG, but mass trend present"
    elif abs(sparc_scatter - tng_scatter) / max(sparc_scatter, tng_scatter) < 0.2:
        verdict = "NON-DISCRIMINATING — SPARC and TNG have similar c_s² scatter"
    else:
        verdict = "INCONCLUSIVE — need more data or better method"

    text_lines.append(f"VERDICT: {verdict}")

    ax.text(0.02, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=8.5, verticalalignment='top')

    fig_path = os.path.join(FIG_DIR, 'gpp_universality_3phase.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return verdict


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    print("=" * 70)
    print("GPP UNIVERSALITY RE-TEST")
    print("SPARC Stabilized + Forward Model + TNG Null Test")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    pts = load_rar_points()
    print(f"  RAR points: {len(pts['galaxy'])}")

    sparc_meta = load_sparc_meta()
    print(f"  SPARC metadata: {len(sparc_meta)} galaxies")

    tng = load_tng()
    print(f"  TNG profiles: {len(tng['galaxy_ids'])} galaxies")

    # Phase A
    phase_a_res = phase_a(pts, sparc_meta)

    # Phase B
    phase_b_res = phase_b(pts, sparc_meta)

    # Phase C
    phase_c_res = phase_c(tng)

    # Plots
    verdict = make_plots(phase_a_res, phase_b_res, phase_c_res)

    # Build summary JSON
    elapsed = time.time() - t0

    # Clean up non-serializable items
    def clean_for_json(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue  # skip arrays
            if isinstance(v, dict):
                # Skip profiles dict (contains arrays)
                if k == 'profiles':
                    continue
                out[k] = clean_for_json(v)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    summary = {
        'test': 'GPP_universality_3phase',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(elapsed, 1),
        'prior_run_reference': {
            'scatter_dex': 0.623,
            'mass_slope': 0.0291,
            'mass_slope_p': 0.032,
            'boson_mass_eV': 3.95e-24,
            'n_galaxies': 627,
        },
        'phase_A_inversion': {
            'HQ': clean_for_json(phase_a_res['HQ']),
            'relaxed': clean_for_json(phase_a_res['relaxed']),
        },
        'phase_B_forward': clean_for_json(phase_b_res),
        'phase_C_TNG_null': clean_for_json(phase_c_res),
        'discrimination': {
            'sparc_scatter': phase_a_res['HQ']['cs2_std_dex'],
            'tng_scatter': phase_c_res['cs2_std_dex'],
            'scatter_ratio': round(phase_a_res['HQ']['cs2_std_dex'] / max(phase_c_res['cs2_std_dex'], 0.001), 3),
            'sparc_mass_slope_p': phase_a_res['HQ']['mass_slope_p'],
            'tng_mass_slope_p': phase_c_res['mass_slope_halo_p'],
        },
        'verdict': verdict,
    }

    json_path = os.path.join(RESULTS_DIR, 'summary_gpp_universality.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {json_path}")

    print(f"\n  Total runtime: {elapsed:.0f}s")
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    summary = main()
