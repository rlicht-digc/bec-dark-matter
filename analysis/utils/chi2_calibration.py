#!/usr/bin/env python3
"""
Utilities for intrinsic-scatter calibration in chi-squared space.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq


def _to_finite_arrays(resid: Sequence[float], sigma: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    resid_arr = np.asarray(resid, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    mask = np.isfinite(resid_arr) & np.isfinite(sigma_arr) & (sigma_arr > 0)
    return resid_arr[mask], sigma_arr[mask]


def reduced_chi2(resid: Sequence[float], sigma: Sequence[float], dof: int) -> float:
    """
    Reduced chi-squared from residuals and uncertainties in the same space.
    """
    if dof <= 0:
        return np.nan
    resid_use, sigma_use = _to_finite_arrays(resid, sigma)
    if resid_use.size == 0:
        return np.nan
    return float(np.sum((resid_use / sigma_use) ** 2) / float(dof))


def chi2_red_given_sigma_int(
    resid: Sequence[float],
    sigma_meas: Sequence[float],
    dof: int,
    sigma_int: float,
) -> float:
    """
    Reduced chi-squared after adding intrinsic scatter in quadrature.
    """
    if not np.isfinite(sigma_int) or sigma_int < 0:
        return np.nan
    sigma_meas_arr = np.asarray(sigma_meas, dtype=float)
    sigma_tot = np.sqrt(np.square(sigma_meas_arr) + sigma_int**2)
    return reduced_chi2(resid, sigma_tot, dof)


def solve_sigma_int_for_chi2_1(
    resid: Sequence[float],
    sigma_meas: Sequence[float],
    dof: int,
    bracket: Optional[Tuple[float, float]] = None,
    tol: float = 1e-6,
) -> Dict[str, object]:
    """
    Solve for sigma_int where reduced chi-squared is ~1.

    Returns dict with:
      sigma_int_best, chi2_red_uncal, chi2_red_cal, method, bracket_used, n_used
    Plus:
      reason (None when successful)
    """
    resid_use, sigma_use = _to_finite_arrays(resid, sigma_meas)
    n_used = int(resid_use.size)

    result: Dict[str, object] = {
        "sigma_int_best": None,
        "chi2_red_uncal": None,
        "chi2_red_cal": None,
        "method": None,
        "bracket_used": None,
        "n_used": n_used,
        "reason": None,
    }

    if dof <= 0:
        result["reason"] = "non_positive_dof"
        return result

    if n_used < 30:
        result["reason"] = "insufficient_finite_points_<30"
        return result

    chi2_uncal = reduced_chi2(resid_use, sigma_use, dof)
    result["chi2_red_uncal"] = float(chi2_uncal) if np.isfinite(chi2_uncal) else None
    if not np.isfinite(chi2_uncal):
        result["reason"] = "non_finite_uncalibrated_chi2"
        return result

    if chi2_uncal <= 1.0:
        result["sigma_int_best"] = 0.0
        result["chi2_red_cal"] = float(chi2_uncal)
        result["method"] = "zero"
        result["bracket_used"] = [0.0, 0.0]
        result["reason"] = "no_inflation_needed"
        return result

    if bracket is None:
        med_sig = float(np.nanmedian(sigma_use)) if sigma_use.size else np.nan
        std_res = float(np.nanstd(resid_use)) if resid_use.size else np.nan
        smax = max(10.0 * med_sig if np.isfinite(med_sig) else 0.0,
                   10.0 * std_res if np.isfinite(std_res) else 0.0,
                   1e-6)
        bracket_used = (0.0, smax)
    else:
        lo, hi = float(bracket[0]), float(bracket[1])
        lo = max(0.0, lo)
        hi = max(lo + 1e-12, hi)
        bracket_used = (lo, hi)
    result["bracket_used"] = [float(bracket_used[0]), float(bracket_used[1])]

    def f_sigma(s: float) -> float:
        val = chi2_red_given_sigma_int(resid_use, sigma_use, dof, s)
        return float(val - 1.0) if np.isfinite(val) else np.nan

    f0 = f_sigma(bracket_used[0])
    f1 = f_sigma(bracket_used[1])

    if np.isfinite(f0) and np.isfinite(f1) and (f0 * f1 <= 0.0):
        try:
            sigma_best = float(brentq(f_sigma, bracket_used[0], bracket_used[1], xtol=tol))
            chi2_cal = chi2_red_given_sigma_int(resid_use, sigma_use, dof, sigma_best)
            result["sigma_int_best"] = sigma_best
            result["chi2_red_cal"] = float(chi2_cal) if np.isfinite(chi2_cal) else None
            result["method"] = "brentq"
            return result
        except Exception:
            # Fall through to min-abs fallback
            pass

    smax = bracket_used[1]
    smin = max(smax * 1e-9, 1e-12)
    grid = np.concatenate([[0.0], np.logspace(np.log10(smin), np.log10(smax), 512)])
    chi_vals = np.array([chi2_red_given_sigma_int(resid_use, sigma_use, dof, s) for s in grid], dtype=float)
    valid = np.isfinite(chi_vals)

    if not np.any(valid):
        result["reason"] = "non_finite_grid_search"
        return result

    abs_diff = np.abs(chi_vals[valid] - 1.0)
    idx_best = int(np.argmin(abs_diff))
    sigma_best = float(grid[valid][idx_best])
    chi2_cal = float(chi_vals[valid][idx_best])

    result["sigma_int_best"] = sigma_best
    result["chi2_red_cal"] = chi2_cal
    result["method"] = "minabs"
    result["reason"] = "no_bracket_crossing_used_minabs"
    return result


__all__ = [
    "reduced_chi2",
    "chi2_red_given_sigma_int",
    "solve_sigma_int_for_chi2_1",
]
