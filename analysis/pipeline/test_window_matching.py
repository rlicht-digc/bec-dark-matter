#!/usr/bin/env python3
"""
TASK 2: Window Matching (C2) — Galaxy-Adapted 10-Point Sampling
================================================================

For each of the 67 SPARC galaxies (≥15 points, Q≤2, Inc 30°–85°):

  1. Compute RAR residuals at full resolution (same as D3 baseline)
  2. Determine R_half from SPARC catalog
  3. Create 10 log-spaced radii in [0.5, 5] × R_half (galaxy-adapted window)
  4. Interpolate detrended residuals onto these 10 radii
  5. Run Lomb-Scargle + permutation null (p<0.05 threshold)

Compare periodic fractions:
  - Full SPARC (67 gal, all points):  37.3%
  - EAGLE fixed (8,837 gal, 10 pts):   2.8%
  - This test (67 gal, 10 pts, adapted window): ???

This tells us whether SPARC's periodicity survives when both the number
of points AND the radial window are matched to a simulation-like protocol.

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

import os
import json
import time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:.0f}s]"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Constants
g_dagger = 1.20e-10
kpc_m = 3.086e19
MIN_POINTS = 15       # SPARC quality cut
N_WINDOW_PTS = 10     # Number of resampled radii (mimicking EAGLE)
R_HALF_LO = 0.5       # Lower bound in units of R_half
R_HALF_HI = 5.0       # Upper bound in units of R_half
N_SURR = 200          # Permutation surrogates per galaxy
PERM_ALPHA = 0.05     # Detection threshold
MIN_RESAMP_PTS = 5    # Minimum interpolated points for LS to be meaningful

np.random.seed(42)
perm_rng = np.random.default_rng(789)

print("=" * 76)
print("TASK 2: WINDOW MATCHING (C2)")
print("  Galaxy-Adapted 10-Point Log-Spaced Sampling")
print("=" * 76)
print(f"  Window: [{R_HALF_LO}, {R_HALF_HI}] × R_half per galaxy")
print(f"  Sampling: {N_WINDOW_PTS} log-spaced radii")
print(f"  LS surrogates: {N_SURR}, detection threshold: p < {PERM_ALPHA}")


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def run_ls_detection(R, resid, n_surr=200, rng=None):
    """
    Run Lomb-Scargle + permutation test.
    Returns (detected, wl_peak, power_peak, p_val).
    """
    n = len(R)
    if n < MIN_RESAMP_PTS:
        return False, np.nan, np.nan, 1.0

    std_r = np.std(resid)
    if std_r < 1e-30:
        return False, np.nan, np.nan, 1.0

    y = (resid - np.mean(resid)) / std_r
    R_extent = R[-1] - R[0]
    if R_extent <= 0:
        return False, np.nan, np.nan, 1.0

    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    if f_max <= f_min:
        return False, np.nan, np.nan, 1.0

    n_freq = min(500, max(50, 10 * n))
    freq_grid = np.linspace(f_min, f_max, n_freq)

    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    f_peak = freq_grid[idx_peak]
    power_peak = float(power[idx_peak])
    wl_peak = 1.0 / f_peak if f_peak > 0 else np.nan

    if rng is None:
        rng = perm_rng

    null_peaks = np.zeros(n_surr)
    for s in range(n_surr):
        y_shuf = rng.permutation(y)
        ls_null = LombScargle(R, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_grid))

    p_val = float(np.mean(null_peaks >= power_peak))
    p_val = max(p_val, 1.0 / (n_surr + 1))
    detected = p_val < PERM_ALPHA

    return detected, float(wl_peak), power_peak, p_val


# ══════════════════════════════════════════════════════════════════
#  PHASE 1: LOAD SPARC DATA
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 1] Loading SPARC data...")

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
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in rc_data[name]:
        rc_data[name][key] = np.array(rc_data[name][key])

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
        sparc_props[name] = {
            'Inc': float(parts[4]),
            'Reff': float(parts[8]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

print(f"  Loaded {len(rc_data)} rotation curves, {len(sparc_props)} property entries")


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: FULL-RESOLUTION BASELINE (reproduce 37.3%)
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 2] Full-resolution SPARC baseline (D3 reference)...")

galaxy_data = []

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
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
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]
    n = len(R_sorted)

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)

    # Spline detrending (same as D3)
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    # Full-resolution LS
    det_full, wl_full, pp_full, p_full = run_ls_detection(R_sorted, eps_det)

    galaxy_data.append({
        'name': name,
        'Reff': prop['Reff'],
        'Vflat': prop['Vflat'],
        'n_pts': n,
        'R': R_sorted,
        'eps_det': eps_det,
        'R_min': float(R_sorted[0]),
        'R_max': float(R_sorted[-1]),
        'R_extent': float(R_sorted[-1] - R_sorted[0]),
        # Full-resolution results
        'full_detected': det_full,
        'full_wl': wl_full,
        'full_power': pp_full,
        'full_p': p_full,
    })

n_galaxies = len(galaxy_data)
n_full_periodic = sum(1 for g in galaxy_data if g['full_detected'])
frac_full = n_full_periodic / n_galaxies

print(f"  Galaxies passing cuts: {n_galaxies}")
print(f"  Full-resolution periodic (p<{PERM_ALPHA}): {n_full_periodic}/{n_galaxies} "
      f"({frac_full:.1%})")


# ══════════════════════════════════════════════════════════════════
#  PHASE 3: GALAXY-ADAPTED WINDOW MATCHING
#   10 log-spaced radii in [0.5, 5] × R_half
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 3] Galaxy-adapted window matching...")
print(f"  Window: [{R_HALF_LO}, {R_HALF_HI}] × R_half")
print(f"  Points: {N_WINDOW_PTS} log-spaced")

window_results = []
n_skipped_reff = 0
n_skipped_coverage = 0

for g in galaxy_data:
    Reff = g['Reff']
    name = g['name']
    R = g['R']
    eps = g['eps_det']

    # Skip if R_eff is invalid
    if Reff <= 0:
        n_skipped_reff += 1
        window_results.append({
            'name': name,
            'skip_reason': 'invalid_Reff',
            'Reff': Reff,
        })
        continue

    # Galaxy-adapted window
    r_lo = R_HALF_LO * Reff
    r_hi = R_HALF_HI * Reff

    # 10 log-spaced target radii
    target_radii = np.logspace(np.log10(r_lo), np.log10(r_hi), N_WINDOW_PTS)

    # Only keep target radii within the galaxy's observed range
    r_obs_min = R[0]
    r_obs_max = R[-1]
    usable = (target_radii >= r_obs_min) & (target_radii <= r_obs_max)
    target_usable = target_radii[usable]

    if len(target_usable) < MIN_RESAMP_PTS:
        n_skipped_coverage += 1
        window_results.append({
            'name': name,
            'skip_reason': 'insufficient_coverage',
            'Reff': round(Reff, 3),
            'window_kpc': [round(r_lo, 2), round(r_hi, 2)],
            'obs_range_kpc': [round(r_obs_min, 2), round(r_obs_max, 2)],
            'n_usable': int(len(target_usable)),
        })
        continue

    # Interpolate detrended residuals onto target radii
    interp_func = interp1d(R, eps, kind='linear', fill_value='extrapolate')
    resid_resampled = interp_func(target_usable)

    # Run LS detection
    det, wl_rec, pp_rec, p_rec = run_ls_detection(target_usable, resid_resampled)

    window_results.append({
        'name': name,
        'skip_reason': None,
        'Reff': round(Reff, 3),
        'window_kpc': [round(r_lo, 2), round(r_hi, 2)],
        'obs_range_kpc': [round(r_obs_min, 2), round(r_obs_max, 2)],
        'n_target': N_WINDOW_PTS,
        'n_usable': int(len(target_usable)),
        'target_radii_kpc': [round(r, 3) for r in target_usable],
        'full_detected': g['full_detected'],
        'full_wl': round(g['full_wl'], 2) if np.isfinite(g['full_wl']) else None,
        'full_p': round(g['full_p'], 4),
        'window_detected': det,
        'window_wl': round(wl_rec, 2) if np.isfinite(wl_rec) else None,
        'window_power': round(pp_rec, 4) if np.isfinite(pp_rec) else None,
        'window_p': round(p_rec, 4),
    })

# Filter to valid results
valid_results = [r for r in window_results if r['skip_reason'] is None]
skipped_results = [r for r in window_results if r['skip_reason'] is not None]

n_valid = len(valid_results)
n_window_periodic = sum(1 for r in valid_results if r['window_detected'])
frac_window = n_window_periodic / n_valid if n_valid > 0 else 0

# Among those that were periodic at full resolution
n_orig_per_in_valid = sum(1 for r in valid_results if r['full_detected'])
n_per_recovered = sum(1 for r in valid_results
                      if r['full_detected'] and r['window_detected'])
recovery_rate = n_per_recovered / n_orig_per_in_valid if n_orig_per_in_valid > 0 else 0

# Among non-periodic, how many become "periodic" (false positives in resampled)
n_orig_nper_in_valid = sum(1 for r in valid_results if not r['full_detected'])
n_nper_flagged = sum(1 for r in valid_results
                     if not r['full_detected'] and r['window_detected'])
false_pos_rate = n_nper_flagged / n_orig_nper_in_valid if n_orig_nper_in_valid > 0 else 0

print(f"\n  Galaxies analyzed: {n_valid}")
print(f"  Skipped (invalid R_eff): {n_skipped_reff}")
print(f"  Skipped (insufficient coverage): {n_skipped_coverage}")
print(f"\n  PERIODIC FRACTION (window-matched, 10 log-spaced pts):")
print(f"    {n_window_periodic}/{n_valid} = {frac_window:.1%}")
print(f"\n  Comparison:")
print(f"    Full SPARC (all points):       {frac_full:.1%}  (reference: 37.3%)")
print(f"    Galaxy-adapted window (10pts): {frac_window:.1%}")
print(f"    EAGLE fixed (10pts, 1-100kpc): 2.8%")
print(f"\n  Recovery of originally periodic galaxies:")
print(f"    {n_per_recovered}/{n_orig_per_in_valid} recovered "
      f"({recovery_rate:.1%})")
print(f"  False positives from non-periodic:")
print(f"    {n_nper_flagged}/{n_orig_nper_in_valid} "
      f"({false_pos_rate:.1%})")

# Median usable points
usable_pts = [r['n_usable'] for r in valid_results]
print(f"\n  Usable points: median {np.median(usable_pts):.0f}, "
      f"range [{min(usable_pts)}, {max(usable_pts)}]")


# ══════════════════════════════════════════════════════════════════
#  PHASE 4: STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 4] Statistical tests...")

# Binomial test: is window fraction significantly above EAGLE's 2.8%?
binom_vs_eagle = stats.binomtest(
    n_window_periodic, n_valid, 0.028, alternative='greater'
)
print(f"  Binomial test (window fraction vs EAGLE 2.8%):")
print(f"    p = {binom_vs_eagle.pvalue:.4e}")

# Binomial test: is window fraction significantly below full SPARC's 37.3%?
binom_vs_full = stats.binomtest(
    n_window_periodic, n_valid, 0.373, alternative='less'
)
print(f"  Binomial test (window fraction vs full SPARC 37.3%):")
print(f"    p = {binom_vs_full.pvalue:.4e}")

# McNemar test: paired comparison (same galaxies, full vs window)
# Only for galaxies present in both analyses
n_both_det = sum(1 for r in valid_results
                 if r['full_detected'] and r['window_detected'])
n_full_only = sum(1 for r in valid_results
                  if r['full_detected'] and not r['window_detected'])
n_window_only = sum(1 for r in valid_results
                    if not r['full_detected'] and r['window_detected'])
n_neither = sum(1 for r in valid_results
                if not r['full_detected'] and not r['window_detected'])

print(f"\n  McNemar contingency (paired, same galaxies):")
print(f"    Both detected:     {n_both_det}")
print(f"    Full only:         {n_full_only}")
print(f"    Window only:       {n_window_only}")
print(f"    Neither:           {n_neither}")

# McNemar's test (exact, mid-p)
if n_full_only + n_window_only > 0:
    mcnemar_p = stats.binomtest(
        n_full_only, n_full_only + n_window_only, 0.5
    ).pvalue
    print(f"    McNemar p (exact): {mcnemar_p:.4e}")
else:
    mcnemar_p = 1.0
    print(f"    McNemar: no discordant pairs")

# p-value comparison: distribution of window p-values
window_pvals = np.array([r['window_p'] for r in valid_results])
full_pvals = np.array([r['full_p'] for r in valid_results])

# Wilcoxon signed-rank on paired p-values
if len(window_pvals) > 5:
    wilcox_stat, wilcox_p = stats.wilcoxon(full_pvals, window_pvals,
                                            alternative='less')
    print(f"\n  Wilcoxon signed-rank (full p < window p?):")
    print(f"    stat = {wilcox_stat:.1f}, p = {wilcox_p:.4e}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 5: WAVELENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 5] Wavelength analysis...")

# Detected galaxies in window
win_det = [r for r in valid_results if r['window_detected']]
full_det = [r for r in valid_results if r['full_detected']]

if win_det:
    win_wl = np.array([r['window_wl'] for r in win_det if r['window_wl'] is not None])
    full_wl_of_windet = np.array([r['full_wl'] for r in win_det
                                   if r['full_wl'] is not None])
    if len(win_wl) > 0:
        print(f"  Window-detected galaxies ({len(win_det)}):")
        print(f"    Window λ: median={np.median(win_wl):.2f} kpc")
        if len(win_wl) >= 4:
            q25, q75 = np.percentile(win_wl, [25, 75])
            print(f"    Window λ: IQR [{q25:.2f}, {q75:.2f}] kpc")

if full_det:
    full_wl_arr = np.array([r['full_wl'] for r in full_det if r['full_wl'] is not None])
    if len(full_wl_arr) > 0:
        print(f"  Full-resolution periodic ({len(full_det)}):")
        print(f"    Full λ: median={np.median(full_wl_arr):.2f} kpc")
        if len(full_wl_arr) >= 4:
            q25, q75 = np.percentile(full_wl_arr, [25, 75])
            print(f"    Full λ: IQR [{q25:.2f}, {q75:.2f}] kpc")


# ══════════════════════════════════════════════════════════════════
#  PHASE 6: PER-GALAXY DETAIL TABLE
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 6] Per-galaxy detail...")

print(f"\n  {'Galaxy':<14} {'Reff':>5} {'Window':>12} {'N_use':>5} "
      f"{'Full_p':>7} {'Win_p':>7} {'Full':>5} {'Win':>5}")
print(f"  {'-'*13} {'-'*5} {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*5} {'-'*5}")

for r in sorted(valid_results, key=lambda x: x['Reff']):
    win_lo, win_hi = r['window_kpc']
    full_flag = "YES" if r['full_detected'] else ""
    win_flag = "YES" if r['window_detected'] else ""
    print(f"  {r['name']:<14} {r['Reff']:5.2f} "
          f"[{win_lo:5.1f},{win_hi:5.1f}] {r['n_usable']:5d} "
          f"{r['full_p']:7.4f} {r['window_p']:7.4f} "
          f"{full_flag:>5} {win_flag:>5}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 7: VERDICT
# ══════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 7] Verdict...")
print("\n" + "=" * 76)
print("VERDICT — WINDOW MATCHING (C2)")
print("=" * 76)

print(f"\n  PERIODIC FRACTIONS:")
print(f"    Full SPARC (reference):       {n_full_periodic}/{n_galaxies} = {frac_full:.1%}")
print(f"    Window-matched (10 log pts):  {n_window_periodic}/{n_valid} = {frac_window:.1%}")
print(f"    EAGLE fixed (reference):      ~2.8%")

if frac_window > 0.25:
    verdict = ("PERIODICITY_SURVIVES — Even with only ~10 points in a galaxy-adapted "
               "window, SPARC shows substantial periodicity. The signal is robust to "
               "simulation-like sampling. EAGLE's 2.8% cannot be explained by sampling alone.")
elif frac_window > 0.10:
    verdict = ("PERIODICITY_REDUCED_BUT_SIGNIFICANT — Window matching reduces periodicity "
               "but it remains well above EAGLE's 2.8%. The SPARC signal is partially "
               "robust to coarse sampling.")
elif frac_window > 0.05:
    verdict = ("PERIODICITY_MARGINAL — Window matching reduces periodicity to near-null "
               "levels. Difficult to distinguish from noise at this sampling density, "
               "but still above EAGLE's 2.8%.")
else:
    verdict = ("PERIODICITY_COLLAPSES — At 10 log-spaced points, SPARC's periodicity "
               "vanishes, matching EAGLE's ~2.8%. The D3 signal may be a resolution "
               "artifact, or the adapted window loses critical outer radii.")

print(f"\n  {verdict}")

# Interpret EAGLE comparison
if frac_window > 3 * 0.028:
    eagle_interp = ("SPARC still shows >3× EAGLE's periodic fraction even under matched "
                    "sampling, supporting a genuine physical difference.")
elif frac_window > 0.028:
    eagle_interp = ("SPARC's periodic fraction drops toward EAGLE's level. The original "
                    "37.3% vs 2.8% gap narrows substantially under matched conditions.")
else:
    eagle_interp = ("Under matched conditions, SPARC and EAGLE periodic fractions converge. "
                    "The original gap was likely dominated by sampling differences.")

print(f"\n  EAGLE comparison: {eagle_interp}")


# ══════════════════════════════════════════════════════════════════
#  SAVE JSON
# ══════════════════════════════════════════════════════════════════
summary = {
    'test': 'window_matching_C2',
    'description': (
        'Galaxy-adapted window matching: resample 67 SPARC galaxies onto '
        '~10 log-spaced radii in [0.5, 5] × R_half, run Lomb-Scargle periodicity '
        'detection. Tests whether SPARC periodicity survives simulation-like sampling.'
    ),
    'parameters': {
        'n_window_pts': N_WINDOW_PTS,
        'r_half_range': [R_HALF_LO, R_HALF_HI],
        'spacing': 'log',
        'n_surrogates': N_SURR,
        'perm_alpha': PERM_ALPHA,
        'min_resamp_pts': MIN_RESAMP_PTS,
        'min_sparc_pts': MIN_POINTS,
    },
    'sample': {
        'n_sparc_galaxies': n_galaxies,
        'n_valid_window': n_valid,
        'n_skipped_reff': n_skipped_reff,
        'n_skipped_coverage': n_skipped_coverage,
        'median_usable_pts': round(float(np.median(usable_pts)), 1),
        'usable_pts_range': [int(min(usable_pts)), int(max(usable_pts))],
    },
    'periodic_fractions': {
        'full_sparc': {
            'n_periodic': n_full_periodic,
            'n_total': n_galaxies,
            'fraction': round(frac_full, 4),
            'label': 'Full SPARC (all points, ≥15 per galaxy)',
        },
        'window_matched': {
            'n_periodic': n_window_periodic,
            'n_total': n_valid,
            'fraction': round(frac_window, 4),
            'label': f'{N_WINDOW_PTS} log-spaced pts in [0.5, 5] × R_half',
        },
        'eagle_reference': {
            'n_periodic': 251,
            'n_total': 8837,
            'fraction': 0.0284,
            'label': 'EAGLE 10 fixed radii (1-100 kpc)',
        },
    },
    'recovery_analysis': {
        'n_orig_periodic_in_valid': n_orig_per_in_valid,
        'n_recovered': n_per_recovered,
        'recovery_rate': round(recovery_rate, 4),
        'n_orig_nonperiodic_in_valid': n_orig_nper_in_valid,
        'n_false_positives': n_nper_flagged,
        'false_positive_rate': round(false_pos_rate, 4),
    },
    'statistical_tests': {
        'binomial_vs_eagle': {
            'test': 'binom(n_window_periodic, n_valid, 0.028, greater)',
            'p_value': float(binom_vs_eagle.pvalue),
            'interpretation': 'Is window fraction significantly above EAGLE 2.8%?',
        },
        'binomial_vs_full_sparc': {
            'test': 'binom(n_window_periodic, n_valid, 0.373, less)',
            'p_value': float(binom_vs_full.pvalue),
            'interpretation': 'Is window fraction significantly below full SPARC 37.3%?',
        },
        'mcnemar_paired': {
            'both_detected': n_both_det,
            'full_only': n_full_only,
            'window_only': n_window_only,
            'neither': n_neither,
            'p_value': float(mcnemar_p),
        },
    },
    'wavelength': {},
    'verdict': verdict,
    'eagle_interpretation': eagle_interp,
    'per_galaxy': valid_results,
    'skipped_galaxies': skipped_results,
}

# Add wavelength stats if available
if win_det:
    win_wl = np.array([r['window_wl'] for r in win_det if r['window_wl'] is not None])
    if len(win_wl) > 0:
        summary['wavelength']['window_detected'] = {
            'n': int(len(win_wl)),
            'median_kpc': round(float(np.median(win_wl)), 2),
            'q25_kpc': round(float(np.percentile(win_wl, 25)), 2) if len(win_wl) >= 4 else None,
            'q75_kpc': round(float(np.percentile(win_wl, 75)), 2) if len(win_wl) >= 4 else None,
        }

if full_det:
    full_wl_arr = np.array([r['full_wl'] for r in full_det if r['full_wl'] is not None])
    if len(full_wl_arr) > 0:
        summary['wavelength']['full_detected'] = {
            'n': int(len(full_wl_arr)),
            'median_kpc': round(float(np.median(full_wl_arr)), 2),
            'q25_kpc': round(float(np.percentile(full_wl_arr, 25)), 2) if len(full_wl_arr) >= 4 else None,
            'q75_kpc': round(float(np.percentile(full_wl_arr, 75)), 2) if len(full_wl_arr) >= 4 else None,
        }

outpath = os.path.join(RESULTS_DIR, 'summary_window_matching.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{elapsed()} Saved: {outpath}")
print("=" * 76)
print("TASK 2 COMPLETE")
print("=" * 76)
