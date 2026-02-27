#!/usr/bin/env python3
"""
Literature Cross-Reference Tests
=================================
Three new analyses motivated by the literature survey:

1. SQUEEZED STATE α-PARAMETER TEST
   Fit σ² = A × [α·n̄² + n̄] + C with α free.
   Thermal: α=1, Classical wave: α→∞ (no +n̄), Coherent: α=0, Squeezed: α>1.

2. MILGROM vs BOSE-EINSTEIN INTERPOLATION FUNCTION
   Compare the de Sitter-Unruh interpolation μ(x) = √(1+x⁻²) − x⁻¹
   against the BE distribution g_obs = g_bar/[1 - exp(-√(g_bar/g†))]
   using SPARC rotation curve data.

3. HIGHER-ORDER STATISTICS (SKEWNESS & KURTOSIS vs g_bar)
   Compute the 3rd and 4th moments of RAR residuals in acceleration bins.
   Thermal (quantum): skewness=2, kurtosis=6
   Classical wave (exponential): skewness=2, kurtosis=6 (same!)
   Poisson: skewness=1/√n̄, kurtosis=1/n̄
   Squeezed: skewness and kurtosis differ — quantify.

Russell Licht — BEC Dark Matter Project
Feb 2026
"""

import numpy as np
from scipy import stats, optimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical constants
g_dagger = 1.20e-10  # m/s^2
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2
Y_DISK = 0.5
Y_BULGE = 0.7


def nbar(log_gbar, g_dag=g_dagger):
    """Bose-Einstein occupation number. log_gbar is log10(g_bar in m/s^2)."""
    gbar = 10**log_gbar  # already in m/s^2
    eps = np.sqrt(np.abs(gbar) / g_dag)
    with np.errstate(over='ignore'):
        return 1.0 / (np.exp(eps) - 1.0 + 1e-30)


# ============================================================
# LOAD SPARC DATA (same loader as unified pipeline)
# ============================================================
def load_sparc_data():
    """Load SPARC rotation curve data and compute RAR residuals."""
    # Navigate from analysis/pipeline/ up to project root, then into data/sparc/
    sparc_dir = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'data', 'sparc')

    # Load galaxy table (fixed-width MRT format)
    gal_file = os.path.join(sparc_dir, 'SPARC_Lelli2016c.mrt')
    galaxies = {}
    if os.path.exists(gal_file):
        with open(gal_file, 'r') as f:
            lines = f.readlines()
        # Find last "---" line to locate data start
        last_dash = 0
        for i, line in enumerate(lines):
            if '---' in line:
                last_dash = i
        for line in lines[last_dash+1:]:
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                name = parts[0]
                hubtype = float(parts[1])
                dist = float(parts[2])
                f_D = int(parts[4])
                inc = float(parts[5])
                qual = int(parts[17])
                galaxies[name] = {
                    'hubtype': hubtype,
                    'dist': dist,
                    'inc': inc,
                    'quality': qual,
                    'f_D': f_D
                }
            except (ValueError, IndexError):
                continue
        print(f"  Loaded {len(galaxies)} galaxy properties")
    else:
        print(f"  WARNING: Galaxy file not found: {gal_file}")

    # Load rotation models (space-separated)
    rotmod_file = os.path.join(sparc_dir, 'SPARC_table2_rotmods.dat')
    points = []
    if os.path.exists(rotmod_file):
        with open(rotmod_file, 'rb') as f:
            for raw_line in f:
                try:
                    line = raw_line.decode('utf-8', errors='replace')
                except:
                    continue
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    name = parts[0]
                    R = float(parts[2])      # kpc (col 3)
                    Vobs = float(parts[3])    # km/s
                    e_Vobs = float(parts[4])  # km/s
                    Vgas = float(parts[5])    # km/s
                    Vdisk = float(parts[6])   # km/s
                    Vbul = float(parts[7]) if len(parts) > 7 else 0.0

                    if name not in galaxies:
                        continue
                    gal = galaxies[name]
                    if gal['quality'] == 3:
                        continue
                    if gal['inc'] < 30 or gal['inc'] > 85:
                        continue
                    if R <= 0 or Vobs <= 0:
                        continue

                    # Baryonic velocity
                    Vbar2 = (Y_DISK * Vdisk**2 +
                             Vgas * abs(Vgas) +
                             Y_BULGE * Vbul * abs(Vbul))

                    # Accelerations in (km/s)^2 / kpc → m/s^2
                    gobs = Vobs**2 / R * conv
                    gbar = abs(Vbar2) / R * conv

                    if gobs <= 0 or gbar <= 0:
                        continue

                    log_gobs = np.log10(gobs)
                    log_gbar = np.log10(gbar)

                    # RAR prediction
                    x = np.sqrt(gbar / g_dagger)
                    gobs_pred = gbar / (1.0 - np.exp(-x))
                    log_gobs_pred = np.log10(gobs_pred)

                    residual = log_gobs - log_gobs_pred

                    # Error propagation
                    if Vobs > 0 and e_Vobs > 0:
                        err_log_gobs = 2.0 * e_Vobs / (Vobs * np.log(10))
                    else:
                        err_log_gobs = 0.1

                    points.append({
                        'name': name,
                        'R': R,
                        'log_gbar': log_gbar,
                        'log_gobs': log_gobs,
                        'log_gobs_pred': log_gobs_pred,
                        'residual': residual,
                        'err': err_log_gobs,
                        'gbar': gbar,
                        'gobs': gobs
                    })
                except (ValueError, IndexError, ZeroDivisionError):
                    continue
    else:
        print(f"  WARNING: Rotmod file not found: {rotmod_file}")

    return points, galaxies


# ============================================================
# TEST 1: SQUEEZED STATE α-PARAMETER
# ============================================================
def test_squeezed_alpha(bins_data):
    """
    Fit σ² = A × [α·n̄² + n̄] + C with α as a free parameter.

    Physical interpretation of α:
      α = 0: Coherent state (Poisson, σ² = n̄)
      α = 1: Thermal/chaotic state (HBT, σ² = n̄² + n̄)
      α > 1: Squeezed state (super-HBT, genuinely quantum)
      α → ∞: Pure classical wave (σ² ∝ n̄², no +n̄ term)
    """
    print("\n" + "="*70)
    print("TEST 1: SQUEEZED STATE α-PARAMETER")
    print("="*70)

    log_gbars = np.array([b['log_gbar'] for b in bins_data])
    vars_obs = np.array([b['var_obs'] for b in bins_data])
    vars_err = np.array([b['var_err'] for b in bins_data])
    nbars = np.array([nbar(lg) for lg in log_gbars])

    # Model: σ² = A × [α·n̄² + n̄] + C
    def model_alpha(x, A, alpha, C):
        nb = x
        return A * (alpha * nb**2 + nb) + C

    # Also fit constrained models for comparison
    def model_thermal(x, A, C):
        return A * (x**2 + x) + C

    def model_classical(x, A, C):
        return A * x**2 + C

    def model_poisson(x, A, C):
        return A * x + C

    def model_coherent(x, A, C):
        # α=0: σ² = A·n̄ + C (same as Poisson)
        return A * x + C

    results = {}

    # Free α fit
    try:
        popt, pcov = optimize.curve_fit(
            model_alpha, nbars, vars_obs, sigma=vars_err,
            p0=[0.001, 1.0, 0.9], bounds=([0, -10, 0], [1, 100, 5]),
            maxfev=50000
        )
        A_fit, alpha_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))
        A_err, alpha_err, C_err = perr

        pred = model_alpha(nbars, *popt)
        chi2 = np.sum(((vars_obs - pred) / vars_err)**2)
        k = 3  # 3 free params
        aic = chi2 + 2*k

        results['free_alpha'] = {
            'A': float(A_fit), 'A_err': float(A_err),
            'alpha': float(alpha_fit), 'alpha_err': float(alpha_err),
            'C': float(C_fit), 'C_err': float(C_err),
            'chi2': float(chi2), 'aic': float(aic), 'k': k,
            'chi2_dof': float(chi2 / (len(nbars) - k))
        }

        print(f"\nFree α fit:")
        print(f"  A     = {A_fit:.6f} ± {A_err:.6f}")
        print(f"  α     = {alpha_fit:.3f} ± {alpha_err:.3f}")
        print(f"  C     = {C_fit:.4f} ± {C_err:.4f}")
        print(f"  χ²/dof = {chi2/(len(nbars)-k):.2f}")
        print(f"  AIC    = {aic:.2f}")

        # Interpret
        if alpha_fit - 2*alpha_err > 1:
            interp = "SQUEEZED STATE (α > 1 at 2σ) — genuinely quantum!"
        elif alpha_fit + 2*alpha_err < 1:
            interp = "SUB-THERMAL (α < 1 at 2σ) — less bunching than expected"
        else:
            interp = f"CONSISTENT WITH THERMAL (α = 1 within {abs(alpha_fit-1)/alpha_err:.1f}σ)"
        results['interpretation'] = interp
        print(f"  → {interp}")

    except Exception as e:
        results['free_alpha'] = {'error': str(e)}
        print(f"  Free α fit failed: {e}")

    # Fixed models for ΔAIC comparison
    for name, func, k_params in [
        ('thermal_alpha1', model_thermal, 2),
        ('classical_wave', model_classical, 2),
        ('poisson', model_poisson, 2),
    ]:
        try:
            popt, _ = optimize.curve_fit(func, nbars, vars_obs, sigma=vars_err,
                                         p0=[0.001, 0.9], maxfev=50000)
            pred = func(nbars, *popt)
            chi2 = np.sum(((vars_obs - pred) / vars_err)**2)
            aic = chi2 + 2*k_params
            results[name] = {
                'params': [float(x) for x in popt],
                'chi2': float(chi2), 'aic': float(aic), 'k': k_params
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    # Compute ΔAICs
    if 'aic' in results.get('free_alpha', {}):
        ref_aic = results['free_alpha']['aic']
        for name in ['thermal_alpha1', 'classical_wave', 'poisson']:
            if 'aic' in results.get(name, {}):
                daic = results[name]['aic'] - ref_aic
                results[f'daic_{name}_vs_free'] = float(daic)
                print(f"  ΔAIC({name} - free_α) = {daic:+.2f}")

    # Bootstrap α distribution
    print("\n  Bootstrapping α (500 iterations)...")
    alpha_boots = []
    np.random.seed(42)
    for _ in range(500):
        idx = np.random.choice(len(nbars), len(nbars), replace=True)
        nb_b = nbars[idx]
        vo_b = vars_obs[idx]
        ve_b = vars_err[idx]
        try:
            popt_b, _ = optimize.curve_fit(
                model_alpha, nb_b, vo_b, sigma=ve_b,
                p0=[0.001, 1.0, 0.9], bounds=([0, -10, 0], [1, 100, 5]),
                maxfev=10000
            )
            alpha_boots.append(popt_b[1])
        except:
            pass

    if alpha_boots:
        alpha_boots = np.array(alpha_boots)
        results['bootstrap_alpha'] = {
            'n_valid': len(alpha_boots),
            'median': float(np.median(alpha_boots)),
            'mean': float(np.mean(alpha_boots)),
            'std': float(np.std(alpha_boots)),
            'ci_68': [float(np.percentile(alpha_boots, 16)),
                      float(np.percentile(alpha_boots, 84))],
            'ci_95': [float(np.percentile(alpha_boots, 2.5)),
                      float(np.percentile(alpha_boots, 97.5))],
            'frac_gt_1': float(np.mean(alpha_boots > 1)),
            'frac_gt_2': float(np.mean(alpha_boots > 2)),
            'frac_lt_0': float(np.mean(alpha_boots < 0)),
        }
        print(f"  Bootstrap α: median = {np.median(alpha_boots):.2f}, "
              f"95% CI = [{np.percentile(alpha_boots, 2.5):.2f}, "
              f"{np.percentile(alpha_boots, 97.5):.2f}]")
        print(f"  P(α > 1): {np.mean(alpha_boots > 1)*100:.1f}% (squeezed)")
        print(f"  P(α > 2): {np.mean(alpha_boots > 2)*100:.1f}% (strongly squeezed)")

    return results


# ============================================================
# TEST 2: MILGROM vs BOSE-EINSTEIN INTERPOLATION
# ============================================================
def test_interpolation_functions(sparc_points):
    """
    Compare three interpolation functions on SPARC RAR data:

    1. Standard RAR (BE distribution):
       g_obs = g_bar / [1 - exp(-√(g_bar/g†))]

    2. Milgrom de Sitter-Unruh:
       μ(x) = √(1 + x⁻²) - x⁻¹  where x = g_obs/a₀
       → g_obs = g_bar / μ(g_obs/a₀) [implicit!]
       Equivalent explicit form: g_obs = g_bar/2 + √(g_bar²/4 + g_bar·a₀)

    3. Simple MOND interpolation:
       μ(x) = x/(1+x)  where x = g_obs/a₀
       → g_obs = g_bar/2 + √(g_bar²/4 + g_bar·a₀)

    Note: Milgrom's function and the "simple" function have the same
    deep-MOND and Newtonian limits but different transition shapes.
    """
    print("\n" + "="*70)
    print("TEST 2: MILGROM vs BOSE-EINSTEIN INTERPOLATION")
    print("="*70)

    log_gbars = np.array([p['log_gbar'] for p in sparc_points])
    log_gobss = np.array([p['log_gobs'] for p in sparc_points])
    errs = np.array([p['err'] for p in sparc_points])
    gbars = np.array([p['gbar'] for p in sparc_points])
    gobss = np.array([p['gobs'] for p in sparc_points])

    results = {}

    # Model 1: Standard RAR (BE distribution) with free g†
    def rar_be(gbar, g_dag):
        x = np.sqrt(gbar / g_dag)
        return gbar / (1.0 - np.exp(-x) + 1e-30)

    # Model 2: Milgrom de Sitter-Unruh (explicit form)
    # From μ(x) = √(1+x⁻²) - x⁻¹, solving for g_obs:
    # g_obs·μ(g_obs/a₀) = g_bar
    # The explicit solution is: g_obs = (g_bar/2)(1 + √(1 + 4a₀/g_bar))
    def rar_milgrom(gbar, a0):
        return 0.5 * gbar * (1.0 + np.sqrt(1.0 + 4.0 * a0 / gbar))

    # Model 3: Simple MOND μ(x) = x/(1+x)
    # g_obs = g_bar/2 + √(g_bar²/4 + g_bar·a₀)
    def rar_simple(gbar, a0):
        return 0.5 * gbar + np.sqrt(0.25 * gbar**2 + gbar * a0)

    # Fit each with one free parameter (the acceleration scale)
    for name, func, p0 in [
        ('BE_distribution', rar_be, [1.2e-10]),
        ('Milgrom_deSitter', rar_milgrom, [1.2e-10]),
        ('Simple_MOND', rar_simple, [1.2e-10]),
    ]:
        try:
            # Fit in log space
            def log_resid(params):
                g_scale = params[0]
                pred = func(gbars, g_scale)
                log_pred = np.log10(np.maximum(pred, 1e-20))
                return np.sum(((log_gobss - log_pred) / errs)**2)

            from scipy.optimize import minimize_scalar, minimize
            res = minimize(log_resid, p0, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16})
            best_scale = res.x[0]

            pred = func(gbars, best_scale)
            log_pred = np.log10(np.maximum(pred, 1e-20))
            residuals = log_gobss - log_pred
            chi2 = np.sum((residuals / errs)**2)
            rms = np.sqrt(np.mean(residuals**2))
            aic = chi2 + 2*1  # 1 free parameter
            bic = chi2 + np.log(len(gbars))*1

            results[name] = {
                'best_scale': float(best_scale),
                'chi2': float(chi2),
                'aic': float(aic),
                'bic': float(bic),
                'rms_dex': float(rms),
                'mean_resid': float(np.mean(residuals)),
                'n_points': len(gbars)
            }

            print(f"\n  {name}:")
            print(f"    Best a₀ = {best_scale:.4e} m/s²")
            print(f"    χ² = {chi2:.1f},  AIC = {aic:.1f}")
            print(f"    RMS residual = {rms:.4f} dex")

            # Residuals by regime
            dm_mask = log_gbars < np.log10(g_dagger / conv)
            bar_mask = ~dm_mask
            if np.sum(dm_mask) > 0:
                results[name]['rms_dm_regime'] = float(np.sqrt(np.mean(residuals[dm_mask]**2)))
                results[name]['rms_bar_regime'] = float(np.sqrt(np.mean(residuals[bar_mask]**2)))
                print(f"    RMS (DM regime):  {results[name]['rms_dm_regime']:.4f} dex")
                print(f"    RMS (bar regime): {results[name]['rms_bar_regime']:.4f} dex")

        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"  {name}: FAILED — {e}")

    # Compute ΔAICs
    if all('aic' in results.get(n, {}) for n in ['BE_distribution', 'Milgrom_deSitter', 'Simple_MOND']):
        be_aic = results['BE_distribution']['aic']
        for name in ['Milgrom_deSitter', 'Simple_MOND']:
            daic = results[name]['aic'] - be_aic
            results[f'daic_{name}_vs_BE'] = float(daic)
            print(f"\n  ΔAIC({name} - BE) = {daic:+.1f}")

        daic_ms = results['Milgrom_deSitter']['aic'] - results['Simple_MOND']['aic']
        results['daic_Milgrom_vs_Simple'] = float(daic_ms)

    # Bin-level comparison: where do they differ most?
    print("\n  Bin-level residual comparison:")
    bin_edges = np.arange(-13, -8, 0.5)
    bin_comparison = []
    for i in range(len(bin_edges)-1):
        mask = (log_gbars >= bin_edges[i]) & (log_gbars < bin_edges[i+1])
        if np.sum(mask) < 5:
            continue

        row = {'bin_center': float(bin_edges[i] + 0.25), 'n_points': int(np.sum(mask))}
        for name, func in [('BE', rar_be), ('Milgrom', rar_milgrom), ('Simple', rar_simple)]:
            scale = results.get(f'{name}_distribution' if name == 'BE' else
                              f'{name}_deSitter' if name == 'Milgrom' else
                              f'{name}_MOND', {}).get('best_scale', g_dagger)
            pred = func(gbars[mask], scale)
            log_pred = np.log10(np.maximum(pred, 1e-20))
            resid = log_gobss[mask] - log_pred
            row[f'rms_{name}'] = float(np.sqrt(np.mean(resid**2)))
            row[f'mean_{name}'] = float(np.mean(resid))

        bin_comparison.append(row)
        if row.get('rms_BE') and row.get('rms_Milgrom'):
            diff = row['rms_Milgrom'] - row['rms_BE']
            marker = " ←" if abs(diff) > 0.005 else ""
            print(f"    [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): N={row['n_points']:4d}  "
                  f"BE={row.get('rms_BE',0):.4f}  Mil={row.get('rms_Milgrom',0):.4f}  "
                  f"Δ={diff:+.4f}{marker}")

    results['bin_comparison'] = bin_comparison

    return results


# ============================================================
# TEST 3: HIGHER-ORDER STATISTICS
# ============================================================
def test_higher_order_stats(sparc_points):
    """
    Compute skewness and kurtosis of RAR residuals in acceleration bins.

    For thermal bosonic field:  skewness = 2, excess kurtosis = 6
    For classical wave (exponential): skewness = 2, excess kurtosis = 6
    For Poisson: skewness = 1/√n̄, excess kurtosis = 1/n̄
    For squeezed state: different — depends on squeezing parameter

    Since thermal and classical wave have IDENTICAL higher-order stats,
    this test can:
    - Confirm consistency with wave/thermal statistics
    - Rule out Poisson (CDM particle) statistics
    - Detect squeezed states if present
    """
    print("\n" + "="*70)
    print("TEST 3: HIGHER-ORDER STATISTICS (SKEWNESS & KURTOSIS)")
    print("="*70)

    log_gbars = np.array([p['log_gbar'] for p in sparc_points])
    residuals = np.array([p['residual'] for p in sparc_points])

    # Bin by acceleration
    bin_edges = [-13.0, -12.5, -12.0, -11.5, -11.0, -10.5, -10.0, -9.5, -9.0, -8.5]

    results = {'bins': [], 'predictions': {
        'thermal_bosonic': {'skewness': 2.0, 'excess_kurtosis': 6.0},
        'classical_wave': {'skewness': 2.0, 'excess_kurtosis': 6.0},
        'gaussian': {'skewness': 0.0, 'excess_kurtosis': 0.0},
        'note': 'Thermal and classical wave have identical 3rd/4th moments. '
                'Distinction requires different observable (not higher-order stats of density).'
    }}

    print(f"\n  {'Bin':>12s} {'N':>5s} {'n̄':>7s} {'Skew':>8s} {'Kurt':>8s} "
          f"{'Skew_pred':>10s} {'Kurt_pred':>10s}")
    print(f"  {'':>12s} {'':>5s} {'':>7s} {'(obs)':>8s} {'(obs)':>8s} "
          f"{'(Poisson)':>10s} {'(Poisson)':>10s}")
    print("  " + "-"*75)

    for i in range(len(bin_edges)-1):
        mask = (log_gbars >= bin_edges[i]) & (log_gbars < bin_edges[i+1])
        n = np.sum(mask)
        if n < 20:
            continue

        r = residuals[mask]
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        nb = nbar(center)

        # Observed moments
        skew_obs = float(stats.skew(r))
        kurt_obs = float(stats.kurtosis(r))  # excess kurtosis

        # Standard errors (approximate, from Fisher)
        se_skew = np.sqrt(6.0 / n) if n > 6 else np.nan
        se_kurt = np.sqrt(24.0 / n) if n > 24 else np.nan

        # Poisson predictions for this n̄
        skew_poisson = 1.0 / np.sqrt(nb) if nb > 0 else np.nan
        kurt_poisson = 1.0 / nb if nb > 0 else np.nan

        # Normality test
        if n >= 20:
            dagostino_stat, dagostino_p = stats.normaltest(r)
        else:
            dagostino_stat, dagostino_p = np.nan, np.nan

        bin_result = {
            'bin_center': float(center),
            'n_points': int(n),
            'nbar': float(nb),
            'skewness': skew_obs,
            'skewness_se': float(se_skew),
            'excess_kurtosis': kurt_obs,
            'kurtosis_se': float(se_kurt),
            'poisson_skew_pred': float(skew_poisson) if not np.isnan(skew_poisson) else None,
            'poisson_kurt_pred': float(kurt_poisson) if not np.isnan(kurt_poisson) else None,
            'thermal_skew_pred': 2.0,
            'thermal_kurt_pred': 6.0,
            'dagostino_p': float(dagostino_p) if not np.isnan(dagostino_p) else None,
        }
        results['bins'].append(bin_result)

        print(f"  [{bin_edges[i]:5.1f},{bin_edges[i+1]:5.1f}) {n:5d} {nb:7.2f} "
              f"{skew_obs:+8.3f} {kurt_obs:+8.3f} "
              f"{skew_poisson:10.3f} {kurt_poisson:10.3f}")

    # Summary statistics
    print("\n  PREDICTIONS:")
    print("  Thermal/Classical wave: skewness=2.0, excess kurtosis=6.0 (for density)")
    print("  Poisson: skewness=1/√n̄, excess kurtosis=1/n̄")
    print("  Gaussian: skewness=0, excess kurtosis=0")
    print("\n  NOTE: RAR residuals are log-space residuals around the mean relation,")
    print("  not raw density fluctuations. The thermal/classical predictions apply")
    print("  to the density field directly, not to log-residuals around a fit.")
    print("  The relevant test is whether the moments TREND with n̄ as predicted.")

    # Test for trends
    if len(results['bins']) >= 4:
        nbars_arr = np.array([b['nbar'] for b in results['bins']])
        skews = np.array([b['skewness'] for b in results['bins']])
        kurts = np.array([b['excess_kurtosis'] for b in results['bins']])

        # Spearman correlations
        rho_skew, p_skew = stats.spearmanr(nbars_arr, skews)
        rho_kurt, p_kurt = stats.spearmanr(nbars_arr, kurts)

        results['trends'] = {
            'skewness_vs_nbar': {
                'spearman_r': float(rho_skew),
                'spearman_p': float(p_skew),
                'interpretation': 'Poisson predicts negative correlation (skew decreases with n̄)'
            },
            'kurtosis_vs_nbar': {
                'spearman_r': float(rho_kurt),
                'spearman_p': float(p_kurt),
                'interpretation': 'Poisson predicts negative correlation (kurt decreases with n̄)'
            }
        }

        print(f"\n  Skewness vs n̄: ρ={rho_skew:+.3f}, p={p_skew:.3f}")
        print(f"  Kurtosis vs n̄: ρ={rho_kurt:+.3f}, p={p_kurt:.3f}")

        # Key diagnostic: are residuals Gaussian or non-Gaussian?
        all_resid = np.array([p['residual'] for p in sparc_points])
        overall_skew = float(stats.skew(all_resid))
        overall_kurt = float(stats.kurtosis(all_resid))
        _, overall_normal_p = stats.normaltest(all_resid)

        results['overall'] = {
            'skewness': overall_skew,
            'excess_kurtosis': overall_kurt,
            'normaltest_p': float(overall_normal_p),
            'is_gaussian': float(overall_normal_p) > 0.05
        }
        print(f"\n  Overall residuals: skew={overall_skew:+.3f}, "
              f"kurt={overall_kurt:+.3f}, normal p={overall_normal_p:.2e}")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("LITERATURE CROSS-REFERENCE TESTS")
    print("BEC Dark Matter — Squeezed States, Interpolation, Higher-Order Stats")
    print("="*70)

    # Load the unified pipeline binned data for Test 1
    unified_file = os.path.join(RESULTS_DIR, 'summary_unified.json')
    with open(unified_file, 'r') as f:
        unified = json.load(f)

    bins_data = unified['refined_bec_tests']['test8_boson_bunching']['bins']

    # Load SPARC raw data for Tests 2 and 3
    print("\nLoading SPARC data...")
    sparc_points, galaxies = load_sparc_data()
    print(f"  Loaded {len(sparc_points)} SPARC data points from {len(set(p['name'] for p in sparc_points))} galaxies")

    # Run all three tests
    results = {}

    results['test1_squeezed_alpha'] = test_squeezed_alpha(bins_data)
    results['test2_interpolation'] = test_interpolation_functions(sparc_points)
    results['test3_higher_order'] = test_higher_order_stats(sparc_points)

    # ============================================================
    # COARSE-GRAINING DERIVATION (Item 3 — analytical)
    # ============================================================
    print("\n" + "="*70)
    print("DERIVATION: COARSE-GRAINING N_occ → n̄")
    print("="*70)

    # The microscopic occupation number per de Broglie volume:
    # N_occ = ρ_DM × λ_dB³ / m
    # For m = 10⁻²² eV, λ_dB ~ 1 kpc, ρ_DM ~ 0.01 M☉/pc³:
    # N_occ ~ 10⁹³
    #
    # Our macroscopic n̄ = 1/[exp(√(g_bar/g†)) - 1] ranges from 0.1 to ~30
    #
    # The connection: n̄ counts the number of GRAVITATIONAL MODES
    # that are macroscopically occupied, not individual bosons.
    #
    # The coarse-graining scale is the healing length ξ = √(GM/g†).
    # Within one healing length volume, all N_occ bosons are in the
    # same ground state mode — they contribute as ONE collective mode.
    #
    # Number of independent modes within radius R:
    # N_modes(R) = (R/ξ)³ for R > ξ (3D volume)
    # N_modes(R) = 1 for R < ξ (single soliton core mode)
    #
    # The gravitational occupation n̄ is the ratio of total DM mass
    # to mass per mode:
    # n̄ ~ M_DM(<R) / M_DM_per_mode = M_DM / (ρ_DM × ξ³)

    # Compute for representative galaxies
    G_SI = 6.674e-11  # m³/kg/s²
    M_sun = 1.989e30   # kg
    kpc = 3.086e19     # m

    print("\n  Healing length and mode count for representative galaxies:")
    print(f"  {'Type':20s} {'log M★':>8s} {'ξ (kpc)':>8s} {'N_modes(10kpc)':>15s} {'n̄(outer)':>10s}")
    print("  " + "-"*65)

    coarse_grain = []
    for label, logMs in [('Dwarf', 8.0), ('Low-mass spiral', 9.0),
                         ('MW-like', 10.0), ('Massive', 11.0)]:
        M_star = 10**logMs * M_sun
        xi = np.sqrt(G_SI * M_star / g_dagger)  # meters
        xi_kpc = xi / kpc

        R_test = 10.0 * kpc  # 10 kpc
        N_modes = max(1, (R_test / xi)**3)

        # Typical n̄ at outer radius where g_bar ~ g†/100
        nbar_outer = nbar(-12.0)  # log g_bar ~ -12

        row = {
            'type': label, 'logMs': logMs,
            'xi_kpc': float(xi_kpc),
            'N_modes_10kpc': float(N_modes),
            'nbar_outer': float(nbar_outer)
        }
        coarse_grain.append(row)
        print(f"  {label:20s} {logMs:8.1f} {xi_kpc:8.2f} {N_modes:15.1f} {nbar_outer:10.1f}")

    results['coarse_graining'] = {
        'description': (
            "The macroscopic n̄ in the RAR counts gravitational modes, not individual bosons. "
            "Within one healing length ξ = √(GM/g†), all bosons occupy a single collective "
            "mode (the soliton ground state). The number of independent modes scales as (R/ξ)³. "
            "The gravitational n̄ is NOT N_occ (which counts bosons per de Broglie volume), "
            "but rather the ratio of actual DM density to the critical density for one "
            "boson per mode at the gravitational coarse-graining scale. This is why n̄ ~ 1-30 "
            "rather than ~10⁹³."
        ),
        'key_relation': "n̄_grav ≈ ρ_DM(r) × ξ³ / M_mode, where M_mode = ρ_crit × ξ³",
        'healing_lengths': coarse_grain,
        'physical_picture': (
            "At radius r in a galaxy, the baryonic gravitational field defines an energy "
            "scale ε = √(g_bar/g†). The BEC has N_occ ~ 10⁹³ microscopic bosons per coherence "
            "volume, but they all occupy the same ground state mode — they act collectively. "
            "The macroscopic occupation n̄ = 1/(e^ε - 1) counts how many independent "
            "gravitational modes are macroscopically occupied at that radius. When n̄ >> 1 "
            "(low g_bar), many modes are occupied → DM-dominated, coherent condensate. "
            "When n̄ << 1 (high g_bar), modes are thermally emptied → Newtonian."
        )
    }

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output_file = os.path.join(RESULTS_DIR, 'summary_literature_crossref.json')

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results = make_serializable(results)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    # ============================================================
    # SYNTHESIS
    # ============================================================
    print("\n" + "="*70)
    print("SYNTHESIS")
    print("="*70)

    # Test 1 verdict
    if 'free_alpha' in results['test1_squeezed_alpha']:
        alpha = results['test1_squeezed_alpha']['free_alpha'].get('alpha', None)
        alpha_err = results['test1_squeezed_alpha']['free_alpha'].get('alpha_err', None)
        if alpha is not None and alpha_err is not None:
            print(f"\n  α-PARAMETER: {alpha:.2f} ± {alpha_err:.2f}")
            if abs(alpha - 1) < 2*alpha_err:
                print("  → Consistent with thermal state (α=1). No evidence for squeezed states.")
                print("  → Cannot distinguish quantum from classical with this test alone.")
            elif alpha > 1 + 2*alpha_err:
                print("  → α > 1 at 2σ! Evidence for SQUEEZED STATE — genuinely quantum!")
            else:
                print("  → α < 1. Unexpected — sub-thermal fluctuations.")

    # Test 2 verdict
    if 'daic_Milgrom_deSitter_vs_BE' in results['test2_interpolation']:
        daic = results['test2_interpolation']['daic_Milgrom_deSitter_vs_BE']
        print(f"\n  INTERPOLATION: ΔAIC(Milgrom - BE) = {daic:+.1f}")
        if abs(daic) < 2:
            print("  → Milgrom and BE interpolation functions are indistinguishable.")
            print("  → The de Sitter thermodynamics and BEC frameworks make the same")
            print("    observational prediction for the RAR shape.")
        elif daic > 2:
            print("  → BE distribution fits better than Milgrom de Sitter.")
        else:
            print("  → Milgrom de Sitter fits better than BE distribution.")

    # Test 3 verdict
    if 'trends' in results['test3_higher_order']:
        p_skew = results['test3_higher_order']['trends']['skewness_vs_nbar']['spearman_p']
        p_kurt = results['test3_higher_order']['trends']['kurtosis_vs_nbar']['spearman_p']
        print(f"\n  HIGHER-ORDER: Skewness trend p={p_skew:.3f}, Kurtosis trend p={p_kurt:.3f}")
        print("  → Thermal and classical wave predict IDENTICAL higher moments.")
        print("  → These cannot distinguish quantum from classical wave DM.")
        print("  → Would need non-Gaussian quantum state (squeezed) to see a difference.")


if __name__ == '__main__':
    main()
