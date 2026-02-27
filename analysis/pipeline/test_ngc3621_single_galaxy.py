#!/usr/bin/env python3
"""
test_ngc3621_single_galaxy.py — Can a SINGLE galaxy localize g†?

NGC3621 from MHONGOOSE/Sorgho+2019: 35 radial points, independent telescope
(MeerKAT), log_gbar from -12.2 to -10.3.  This test checks whether one
galaxy's RAR residual profile carries a detectable signature of g†.

Tests:
  1. Load NGC3621 RAR points from mhongoose_rar_all.tsv
  2. Binned scatter (σ) in 5 bins across its g_bar range
  3. Scatter trend: inversion (peak) vs monotonic?
  4. RAR residual profile vs standard RAR function
  5. Residual sign-change localization relative to g† = 10^{-9.92}
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analysis_tools import (
    rar_function, rar_residuals, numerical_derivative,
    find_zero_crossings, g_dagger, LOG_G_DAGGER,
)

# ── paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mhongoose')
TSV_FILE = os.path.join(DATA_DIR, 'mhongoose_rar_all.tsv')


def load_ngc3621():
    """Load NGC3621 rows from the MHONGOOSE RAR file."""
    log_gbar, log_gobs = [], []
    with open(TSV_FILE) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if parts[0] == 'NGC3621':
                log_gbar.append(float(parts[1]))
                log_gobs.append(float(parts[2]))
    return np.array(log_gbar), np.array(log_gobs)


def binned_scatter(log_gbar, residuals, n_bins=5):
    """Compute σ of residuals in equal-width bins.  Returns (centers, sigmas, counts)."""
    lo, hi = log_gbar.min(), log_gbar.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, sigmas, counts = [], [], []
    for j in range(n_bins):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j + 1])
        if j == n_bins - 1:          # include right edge in last bin
            mask |= (log_gbar == edges[j + 1])
        n = mask.sum()
        centers.append(0.5 * (edges[j] + edges[j + 1]))
        counts.append(n)
        sigmas.append(float(np.std(residuals[mask])) if n >= 2 else np.nan)
    return np.array(centers), np.array(sigmas), np.array(counts)


def residual_sign_change(log_gbar, residuals):
    """Find log_gbar values where RAR residuals cross zero (linear interpolation).

    Sort by log_gbar ascending, then find sign changes.
    Returns list of crossing log_gbar values.
    """
    order = np.argsort(log_gbar)
    x = log_gbar[order]
    y = residuals[order]
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            x_cross = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            crossings.append(float(x_cross))
    return crossings


def main():
    print("=" * 70)
    print("TEST: Single-galaxy g† localization — NGC3621 (MHONGOOSE/Sorgho+2019)")
    print("=" * 70)

    # ── 1. Load data ───────────────────────────────────────────
    log_gbar, log_gobs = load_ngc3621()
    N = len(log_gbar)
    print(f"\n[1] NGC3621: {N} radial points")
    print(f"    log_gbar range: [{log_gbar.min():.3f}, {log_gbar.max():.3f}]")
    print(f"    log_gobs range: [{log_gobs.min():.3f}, {log_gobs.max():.3f}]")

    if N == 0:
        print("ERROR: no NGC3621 data found"); sys.exit(1)

    # ── 2. RAR residuals & binned scatter ──────────────────────
    residuals = rar_residuals(log_gbar, log_gobs)
    print(f"\n[2] RAR residuals: mean={residuals.mean():.4f}, "
          f"std={residuals.std():.4f}, range=[{residuals.min():.4f}, {residuals.max():.4f}]")

    N_BINS = 5
    centers, sigmas, counts = binned_scatter(log_gbar, residuals, n_bins=N_BINS)
    print(f"\n    Binned scatter ({N_BINS} bins):")
    print(f"    {'Bin center':>12s}  {'N':>4s}  {'σ (dex)':>8s}")
    for c, s, n in zip(centers, sigmas, counts):
        print(f"    {c:12.3f}  {n:4d}  {s:8.4f}")

    # ── 3. Scatter trend: inversion vs monotonic ───────────────
    valid = ~np.isnan(sigmas)
    valid_centers = centers[valid]
    valid_sigmas = sigmas[valid]

    if len(valid_sigmas) >= 3:
        dsigma = numerical_derivative(valid_centers, valid_sigmas)
        crossings_sigma = find_zero_crossings(valid_centers, dsigma, direction='pos_to_neg')

        # Also check monotonic trend with Spearman rank correlation
        from scipy.stats import spearmanr
        rho, p_mono = spearmanr(valid_centers, valid_sigmas)

        print(f"\n[3] Scatter trend analysis:")
        print(f"    Spearman ρ(center, σ) = {rho:+.3f}  (p = {p_mono:.3f})")
        if crossings_sigma:
            print(f"    dσ/d(log g_bar) zero-crossings (peaks): {crossings_sigma}")
            print(f"    → Inversion (scatter peak) detected")
        else:
            direction = "increasing toward low g_bar" if rho < 0 else "decreasing toward low g_bar"
            print(f"    No inversion peak found — scatter is {direction}")
            print(f"    dσ/d(log g_bar) = {dsigma}")
    else:
        print("\n[3] Too few valid bins for trend analysis")
        crossings_sigma = []

    # ── 4. Residual profile vs standard RAR ────────────────────
    # Sort by log_gbar and show the residual curve
    order = np.argsort(log_gbar)
    sorted_gbar = log_gbar[order]
    sorted_resid = residuals[order]

    print(f"\n[4] Residual profile (sorted by log_gbar, every 5th point):")
    print(f"    {'log_gbar':>10s}  {'log_gobs_pred':>14s}  {'log_gobs_obs':>13s}  {'residual':>9s}")
    predicted = rar_function(sorted_gbar)
    for i in range(0, N, max(1, N // 7)):
        print(f"    {sorted_gbar[i]:10.3f}  {predicted[i]:14.3f}  "
              f"{log_gobs[order][i]:13.3f}  {sorted_resid[i]:+9.4f}")

    # Mean residual in bins (systematic offset)
    print(f"\n    Bin-averaged residuals:")
    lo, hi = log_gbar.min(), log_gbar.max()
    edges = np.linspace(lo, hi, N_BINS + 1)
    for j in range(N_BINS):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j + 1])
        if j == N_BINS - 1:
            mask |= (log_gbar == edges[j + 1])
        if mask.sum() >= 1:
            c = 0.5 * (edges[j] + edges[j + 1])
            print(f"      bin {c:.2f}: <resid> = {residuals[mask].mean():+.4f}  "
                  f"(N={mask.sum()})")

    # ── 5. Residual sign change near g† ────────────────────────
    sign_crossings = residual_sign_change(log_gbar, residuals)
    print(f"\n[5] Residual sign-change analysis:")
    print(f"    g† = 10^{{{LOG_G_DAGGER:.4f}}} m/s²  (log g† = {LOG_G_DAGGER:.4f})")
    print(f"    NGC3621 log_gbar range: [{log_gbar.min():.3f}, {log_gbar.max():.3f}]")

    if sign_crossings:
        print(f"    Sign crossings found at log_gbar = {[f'{x:.3f}' for x in sign_crossings]}")
        nearest = min(sign_crossings, key=lambda x: abs(x - LOG_G_DAGGER))
        delta = nearest - LOG_G_DAGGER
        print(f"    Nearest crossing to g†: log_gbar = {nearest:.4f}  "
              f"(Δ = {delta:+.4f} dex from g†)")

        # Extrapolation: does the residual trend toward zero at g†?
        # Use the 5 highest-gbar points to estimate the extrapolated crossing
        top_n = min(8, N)
        top_mask = np.argsort(log_gbar)[-top_n:]
        from numpy.polynomial import polynomial as P
        coefs = P.polyfit(log_gbar[top_mask], residuals[top_mask], deg=1)
        # polyfit returns [c0, c1] where y = c0 + c1*x
        slope, intercept = coefs[1], coefs[0]
        if abs(slope) > 1e-10:
            extrap_crossing = -intercept / slope
            print(f"\n    Linear extrapolation of high-gbar residuals:")
            print(f"      slope = {slope:.4f}, intercept = {intercept:+.4f}")
            print(f"      extrapolated zero-crossing at log_gbar = {extrap_crossing:.4f}")
            print(f"      distance from g†: {extrap_crossing - LOG_G_DAGGER:+.4f} dex")
    else:
        print("    No residual sign crossings found within the data range")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Determine if residuals are systematically negative near g†
    high_gbar_mask = log_gbar > -10.5
    low_gbar_mask = log_gbar < -11.5
    mid_gbar_mask = (~high_gbar_mask) & (~low_gbar_mask)

    mean_hi = residuals[high_gbar_mask].mean() if high_gbar_mask.sum() > 0 else np.nan
    mean_mid = residuals[mid_gbar_mask].mean() if mid_gbar_mask.sum() > 0 else np.nan
    mean_lo = residuals[low_gbar_mask].mean() if low_gbar_mask.sum() > 0 else np.nan

    print(f"\n  Mean residuals by region:")
    print(f"    High gbar (>{-10.5}):  {mean_hi:+.4f}  (N={high_gbar_mask.sum()}, "
          f"nearest to g†)")
    print(f"    Mid  gbar:             {mean_mid:+.4f}  (N={mid_gbar_mask.sum()})")
    print(f"    Low  gbar (<{-11.5}): {mean_lo:+.4f}  (N={low_gbar_mask.sum()})")

    # Key result
    if sign_crossings:
        nearest = min(sign_crossings, key=lambda x: abs(x - LOG_G_DAGGER))
        delta = abs(nearest - LOG_G_DAGGER)
        if delta < 0.5:
            verdict = (f"YES — residual sign change at {nearest:.3f}, "
                       f"only {delta:.2f} dex from g†")
        else:
            verdict = (f"PARTIAL — sign change at {nearest:.3f}, "
                       f"but {delta:.2f} dex from g†")
    else:
        verdict = "NO — no residual sign change detected"

    print(f"\n  Can NGC3621 alone localize g†?")
    print(f"    {verdict}")

    # Scatter assessment
    if crossings_sigma:
        sc_nearest = min(crossings_sigma, key=lambda x: abs(x - LOG_G_DAGGER))
        print(f"    Scatter peak at: {sc_nearest:.3f}")
    else:
        print(f"    Scatter trend: {'monotonic' if len(valid_sigmas) >= 3 else 'insufficient data'}")

    print(f"\n  Galaxy total σ_RAR = {residuals.std():.4f} dex  "
          f"(cf. SPARC ensemble ~0.13 dex)")
    print()


if __name__ == '__main__':
    main()
