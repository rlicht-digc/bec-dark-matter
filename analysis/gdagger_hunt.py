#!/usr/bin/env python3
"""
gdagger_hunt.py — Systematic search for the BEC acceleration scale g† in
non-RAR contexts.

Tests whether g† ≈ 1.2e-10 m/s² is:
  (A) a RAR-specific scale, OR
  (B) a universal scale reappearing in independent physics.

Components:
  1. Dimensionless-group generator (practical Buckingham-Pi)
  2. Kernel library + matcher (BE-like, coth, logistic, power-law)
  3. Anti-numerology controls (scale injection, shuffle, nearby-scale)
  4. Experiment runner with reproducible timestamped outputs

Consistent with analysis_tools.py patterns from the BEC-DM pipeline.

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from itertools import product as iterproduct
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PHYSICS CONSTANTS (mirroring analysis_tools.py)
# ============================================================

G_DAGGER: float = 1.20e-10          # m/s² — BEC condensation scale
LOG_G_DAGGER: float = np.log10(G_DAGGER)  # -9.9208…

G_SI: float = 6.674e-11             # m³ kg⁻¹ s⁻²
M_SUN: float = 1.989e30             # kg
KPC_M: float = 3.086e19             # m / kpc
C_LIGHT: float = 2.998e8            # m/s
HBAR: float = 1.055e-34             # J·s
K_BOLTZ: float = 1.381e-23          # J/K

# Cosmological constants (Planck 2018)
H0_SI: float = 67.74e3 / 3.086e22  # 1/s  (67.74 km/s/Mpc)
LAMBDA_CC: float = 1.11e-52         # m⁻² (cosmological constant, Λ)

# Derived acceleration scales
A_LAMBDA: float = C_LIGHT**2 * np.sqrt(LAMBDA_CC / 3.0)  # c² √(Λ/3)
A_HUBBLE: float = C_LIGHT * H0_SI                         # cH₀


# ============================================================
# §1  DIMENSIONLESS-GROUP GENERATOR
# ============================================================

@dataclass
class PhysicalQuantity:
    """A named physical quantity with SI dimensions [M, L, T, K]."""
    name: str
    symbol: str
    value: float
    dims: Tuple[float, float, float, float]  # (M, L, T, K)
    description: str = ""


# Standard constants library
STANDARD_CONSTANTS: Dict[str, PhysicalQuantity] = {
    "c":     PhysicalQuantity("speed of light",  "c",  C_LIGHT,
                              (0, 1, -1, 0)),
    "G":     PhysicalQuantity("gravitational constant", "G", G_SI,
                              (-1, 3, -2, 0)),
    "hbar":  PhysicalQuantity("reduced Planck", "ℏ", HBAR,
                              (1, 2, -1, 0)),
    "kB":    PhysicalQuantity("Boltzmann", "k_B", K_BOLTZ,
                              (1, 2, -2, -1)),
    "Lambda": PhysicalQuantity("cosmo. constant", "Λ", LAMBDA_CC,
                               (0, -2, 0, 0)),
    "H0":   PhysicalQuantity("Hubble constant", "H₀", H0_SI,
                              (0, 0, -1, 0)),
}


def generate_pi_groups(
    target_dims: Tuple[float, float, float, float],
    constants: Dict[str, PhysicalQuantity],
    context_vars: Optional[Dict[str, PhysicalQuantity]] = None,
    exponent_set: Sequence[float] = (-2, -1, -0.5, 0, 0.5, 1, 2),
    max_constants: int = 3,
    require_lambda: bool = False,
) -> List[Dict[str, Any]]:
    """Generate dimensionless groups that form a ratio with a target quantity.

    Finds combinations of constants (and optional context variables) whose
    product has the same dimensions as `target_dims`, producing a
    dimensionless ratio  target / (product of constants^exponents).

    Parameters
    ----------
    target_dims : tuple of 4 floats
        SI dimensions [M, L, T, K] of the target quantity.
    constants : dict
        Pool of named PhysicalQuantity objects.
    context_vars : dict or None
        Additional variables (e.g., mass, radius of a specific system).
    exponent_set : sequence of float
        Allowed rational exponents.
    max_constants : int
        Maximum number of constants in a single group.
    require_lambda : bool
        If True, only keep groups that use Λ.

    Returns
    -------
    list of dict
        Each dict: {formula, exponents, numeric_value, uses_lambda,
                    n_constants, rank_score}.
    """
    pool: Dict[str, PhysicalQuantity] = dict(constants)
    if context_vars:
        pool.update(context_vars)

    names = list(pool.keys())
    target = np.array(target_dims, dtype=float)

    results: List[Dict[str, Any]] = []

    # Enumerate combinations of 1..max_constants quantities
    for n_use in range(1, min(max_constants + 1, len(names) + 1)):
        # Use index combinations to avoid order duplicates
        from itertools import combinations
        for idx_combo in combinations(range(len(names)), n_use):
            combo_names = [names[i] for i in idx_combo]
            combo_dims = np.array([pool[n].dims for n in combo_names],
                                  dtype=float)
            combo_vals = np.array([pool[n].value for n in combo_names],
                                  dtype=float)

            # Try all exponent combos
            for exps in iterproduct(
                *([exp for exp in exponent_set if exp != 0]
                  for _ in range(n_use))
            ):
                exp_arr = np.array(exps, dtype=float)
                resultant_dims = exp_arr @ combo_dims  # shape (4,)
                if np.allclose(resultant_dims, target, atol=1e-10):
                    # This combination matches target dimensions
                    numeric = float(np.prod(combo_vals ** exp_arr))
                    uses_lambda = "Lambda" in combo_names
                    n_ctx = sum(1 for n in combo_names
                                if context_vars and n in context_vars)

                    # Build formula string
                    parts = []
                    for nm, ex in zip(combo_names, exps):
                        sym = pool[nm].symbol
                        if ex == 1:
                            parts.append(sym)
                        elif ex == -1:
                            parts.append(f"1/{sym}")
                        elif ex == int(ex):
                            parts.append(f"{sym}^{int(ex)}")
                        else:
                            parts.append(f"{sym}^{ex}")
                    formula = " · ".join(parts)

                    # Rank: prefer Λ-using, then fewer free astro vars
                    rank = 0
                    if uses_lambda:
                        rank += 100
                    rank -= n_ctx * 10     # penalize context variables
                    rank -= n_use          # prefer simpler

                    if not require_lambda or uses_lambda:
                        results.append({
                            "formula": formula,
                            "exponents": {n: float(e)
                                          for n, e in zip(combo_names, exps)},
                            "numeric_value": numeric,
                            "log10_value": float(np.log10(
                                abs(numeric) + 1e-300)),
                            "uses_lambda": uses_lambda,
                            "n_constants": n_use,
                            "n_context_vars": n_ctx,
                            "rank_score": rank,
                        })

    # Deduplicate by numeric value (within 0.01 dex)
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = round(r["log10_value"], 2)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: -r["rank_score"])
    return unique


# ============================================================
# §2  KERNEL LIBRARY + MATCHER
# ============================================================

def _safe_exp(x: NDArray) -> NDArray:
    """Clipped exp to avoid overflow."""
    return np.exp(np.clip(x, -500, 500))


# --- Kernel functions ---
# Each takes (x, scale) → y, where x is the physical variable
# and scale is the free acceleration parameter.

def kernel_be_rar(x: NDArray, scale: float) -> NDArray:
    """BE-RAR kernel: K(x) = 1 / (1 - exp(-√(x/s))).

    This is the RAR mapping form (g_obs/g_bar).
    """
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    denom = 1.0 - _safe_exp(-eps)
    return 1.0 / np.maximum(denom, 1e-30)


def kernel_be_occupation(x: NDArray, scale: float) -> NDArray:
    """Bose-Einstein occupation number: 1/(exp(√(x/s)) - 1)."""
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    return 1.0 / np.maximum(_safe_exp(eps) - 1.0, 1e-30)


def kernel_be_cousin(x: NDArray, scale: float) -> NDArray:
    """BE cousin: ε/(1-exp(-ε)), ε = √(x/s)."""
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    denom = 1.0 - _safe_exp(-eps)
    return eps / np.maximum(denom, 1e-30)


def kernel_coth(x: NDArray, scale: float) -> NDArray:
    """Hyperbolic cotangent: coth(√(x/s))."""
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    return 1.0 / np.tanh(np.maximum(eps, 1e-30))


def kernel_tanh(x: NDArray, scale: float) -> NDArray:
    """Hyperbolic tangent: tanh(√(x/s))."""
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    return np.tanh(eps)


def kernel_logistic(x: NDArray, scale: float) -> NDArray:
    """Logistic: 1/(1 + exp(-√(x/s) + 1))."""
    eps = np.sqrt(np.maximum(x / scale, 1e-30))
    return 1.0 / (1.0 + _safe_exp(-(eps - 1.0)))


def kernel_power_law(x: NDArray, scale: float) -> NDArray:
    """Power-law control: (x/s)^0.5 + 1 (no special scale)."""
    return np.sqrt(np.maximum(x / scale, 1e-30)) + 1.0


# Kernel registry
KERNEL_REGISTRY: Dict[str, Callable] = {
    "BE_RAR":        kernel_be_rar,
    "BE_occupation": kernel_be_occupation,
    "BE_cousin":     kernel_be_cousin,
    "coth":          kernel_coth,
    "tanh":          kernel_tanh,
    "logistic":      kernel_logistic,
    "power_law":     kernel_power_law,
}


@dataclass
class KernelFitResult:
    """Result of fitting one kernel to data."""
    kernel_name: str
    scale_best: float
    log_scale_best: float
    amplitude: float
    offset: float
    residual_rms: float
    aic: float
    bic: float
    cv_rmse: float
    n_params: int
    n_data: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


def _jsonable(v: Any) -> Any:
    """Make value JSON-serializable."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if np.isfinite(v) else str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _fit_kernel_at_scale(
    kernel_fn: Callable,
    x: NDArray,
    y: NDArray,
    scale: float,
    fit_mode: str = "linear",
) -> Tuple[float, float, float]:
    """Fit kernel to data at a fixed scale.

    Modes:
      "linear":  y = A * kernel(x, scale) + B  (2 nuisance params)
      "log":     log(y) = A * log(kernel(x, scale)) + B  (2 nuisance params)
      "direct":  y = kernel(x, scale)  (0 nuisance params — exact identity)

    Returns (amplitude, offset, rms_residual).
    """
    k_vals = kernel_fn(x, scale)
    # Guard against degenerate kernels
    if not np.all(np.isfinite(k_vals)):
        return 0.0, 0.0, 1e30

    n = len(x)

    if fit_mode == "direct":
        # No free parameters — test how well kernel matches y directly
        resid = y - k_vals
        rms = float(np.sqrt(np.mean(resid**2)))
        return 1.0, 0.0, rms

    elif fit_mode == "log":
        # Fit in log space: better for multiplicative data (like RAR)
        valid = (k_vals > 0) & (y > 0)
        if np.sum(valid) < 5:
            return 0.0, 0.0, 1e30
        log_k = np.log10(k_vals[valid])
        log_y = np.log10(y[valid])
        n_v = int(np.sum(valid))

        k_mean = np.mean(log_k)
        y_mean = np.mean(log_y)
        var_k = np.var(log_k)

        if var_k < 1e-30:
            return 0.0, y_mean, float(np.sqrt(np.mean(
                (log_y - y_mean)**2)))

        A = np.sum((log_k - k_mean) * (log_y - y_mean)) / (n_v * var_k)
        B = y_mean - A * k_mean
        resid_log = log_y - (A * log_k + B)
        rms = float(np.sqrt(np.mean(resid_log**2)))
        return float(A), float(B), rms

    else:  # "linear"
        # Linear regression: y = A*k + B
        k_mean = np.mean(k_vals)
        y_mean = np.mean(y)
        var_k = np.var(k_vals)

        if var_k < 1e-30:
            return 0.0, y_mean, float(np.sqrt(np.mean(
                (y - y_mean)**2)))

        A = np.sum((k_vals - k_mean) * (y - y_mean)) / (n * var_k)
        B = y_mean - A * k_mean
        resid = y - (A * k_vals + B)
        rms = float(np.sqrt(np.mean(resid**2)))
        return float(A), float(B), rms


def fit_kernel(
    kernel_name: str,
    x: NDArray,
    y: NDArray,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 200,
    fix_scale: Optional[float] = None,
    n_cv_folds: int = 5,
    rng_seed: int = 42,
    fit_mode: str = "linear",
) -> KernelFitResult:
    """Fit a kernel to (x, y) data, optimizing the acceleration scale.

    Parameters
    ----------
    kernel_name : str
        Key into KERNEL_REGISTRY.
    x, y : arrays
        x is the physical variable (e.g., acceleration in m/s²),
        y is the observed response.
    scale_range : tuple
        (lo, hi) for log-uniform scale search.
    n_grid : int
        Grid points for initial scale scan.
    fix_scale : float or None
        If provided, skip optimization and use this scale.
    n_cv_folds : int
        K-fold cross-validation folds.
    rng_seed : int
        Seed for CV fold assignment.
    fit_mode : str
        "linear" (y=Ak+B), "log" (log y = A log k + B), "direct" (y=k).

    Returns
    -------
    KernelFitResult
    """
    kernel_fn = KERNEL_REGISTRY[kernel_name]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    assert n == len(y), "x and y must have same length"
    assert n >= 10, f"Need at least 10 data points, got {n}"

    if fix_scale is not None:
        # Fixed-scale mode
        A, B, rms = _fit_kernel_at_scale(kernel_fn, x, y, fix_scale,
                                          fit_mode)
        best_scale = fix_scale
    else:
        # Grid search in log space
        log_scales = np.linspace(np.log10(scale_range[0]),
                                 np.log10(scale_range[1]), n_grid)
        best_rms = 1e30
        best_scale = scale_range[0]
        for ls in log_scales:
            s = 10**ls
            _, _, rms_trial = _fit_kernel_at_scale(kernel_fn, x, y, s,
                                                    fit_mode)
            if rms_trial < best_rms:
                best_rms = rms_trial
                best_scale = s

        # Refine with bounded optimization
        def neg_fit(log_s: float) -> float:
            s = 10**log_s
            _, _, rms_trial = _fit_kernel_at_scale(kernel_fn, x, y, s,
                                                    fit_mode)
            return rms_trial

        res = minimize_scalar(
            neg_fit,
            bounds=(np.log10(scale_range[0]), np.log10(scale_range[1])),
            method="bounded",
        )
        if res.success:
            best_scale = 10**res.x

        A, B, rms = _fit_kernel_at_scale(kernel_fn, x, y, best_scale,
                                          fit_mode)

    # AIC / BIC
    n_params_map = {"linear": 2, "log": 2, "direct": 0}
    n_params = n_params_map[fit_mode] + (0 if fix_scale else 1)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * rms**2 + 1e-300) - n / 2
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    # K-fold CV
    rng = np.random.RandomState(rng_seed)
    fold_idx = rng.permutation(n) % n_cv_folds
    cv_errors_sq: List[float] = []

    for fold in range(n_cv_folds):
        train_mask = fold_idx != fold
        test_mask = fold_idx == fold
        if np.sum(train_mask) < 5 or np.sum(test_mask) < 2:
            continue

        x_tr, y_tr = x[train_mask], y[train_mask]
        x_te, y_te = x[test_mask], y[test_mask]

        A_tr, B_tr, _ = _fit_kernel_at_scale(kernel_fn, x_tr, y_tr,
                                              best_scale, fit_mode)
        k_te = kernel_fn(x_te, best_scale)
        if fit_mode == "direct":
            pred_te = k_te
        elif fit_mode == "log":
            valid = (k_te > 0) & (y_te > 0)
            if np.sum(valid) < 2:
                continue
            log_pred = A_tr * np.log10(k_te[valid]) + B_tr
            cv_errors_sq.extend((np.log10(y_te[valid]) - log_pred)**2)
            continue
        else:
            pred_te = A_tr * k_te + B_tr
        cv_errors_sq.extend((y_te - pred_te)**2)

    cv_rmse = float(np.sqrt(np.mean(cv_errors_sq))) if cv_errors_sq else rms

    return KernelFitResult(
        kernel_name=kernel_name,
        scale_best=float(best_scale),
        log_scale_best=float(np.log10(best_scale)),
        amplitude=float(A),
        offset=float(B),
        residual_rms=float(rms),
        aic=float(aic),
        bic=float(bic),
        cv_rmse=float(cv_rmse),
        n_params=n_params,
        n_data=n,
    )


def match_kernels(
    x: NDArray,
    y: NDArray,
    kernel_names: Optional[Sequence[str]] = None,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 200,
    n_cv_folds: int = 5,
    rng_seed: int = 42,
    fit_mode: str = "linear",
) -> List[KernelFitResult]:
    """Fit all kernels and return sorted by AIC (best first).

    Parameters
    ----------
    x, y : arrays
        Physical variable and response.
    kernel_names : list of str or None
        If None, use all registered kernels.
    scale_range, n_grid, n_cv_folds, rng_seed:
        Passed to fit_kernel().
    fit_mode : str
        "linear", "log", or "direct". Use "log" for multiplicative data
        (e.g., RAR mapping ratio), "direct" for identity tests.

    Returns
    -------
    list of KernelFitResult, sorted by AIC ascending.
    """
    if kernel_names is None:
        kernel_names = list(KERNEL_REGISTRY.keys())

    results = []
    for name in kernel_names:
        try:
            r = fit_kernel(name, x, y,
                           scale_range=scale_range, n_grid=n_grid,
                           n_cv_folds=n_cv_folds, rng_seed=rng_seed,
                           fit_mode=fit_mode)
            results.append(r)
        except Exception as exc:
            logging.warning("Kernel %s failed: %s", name, exc)

    # Sort by AIC, with tiebreaker: when AIC within 1.0, prefer
    # (1) lower CV RMSE, (2) scale closer to g†
    results.sort(key=lambda r: (
        round(r.aic, 0),       # Primary: AIC (rounded to break float noise)
        round(r.cv_rmse, 6),   # Tiebreak 1: CV RMSE
        abs(r.log_scale_best - LOG_G_DAGGER),  # Tiebreak 2: proximity to g†
    ))
    return results


# ============================================================
# §3  ANTI-NUMEROLOGY CONTROLS
# ============================================================

@dataclass
class ScaleScanResult:
    """Result of scanning fit quality across acceleration scales."""
    log_scales: List[float]
    rms_values: List[float]
    aic_values: List[float]
    best_log_scale: float
    best_rms: float
    delta_aic_at_gdagger: float   # AIC(g†) - AIC(best)
    peak_sharpness: float         # 2nd derivative of AIC at optimum
    within_0p1_dex: bool          # Is best within ±0.1 dex of g†?
    delta_aic_pm_0p1_dex: Dict[str, float] = field(
        default_factory=dict
    )  # ΔAIC at best ±0.1 dex neighbors: {"minus": ..., "plus": ...}
    delta_aic_pm_0p1_per_dof: Dict[str, float] = field(
        default_factory=dict
    )  # Same, normalized by N (dataset-size-independent)

    def to_dict(self) -> Dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


def scale_injection_scan(
    kernel_name: str,
    x: NDArray,
    y: NDArray,
    log_scale_range: Tuple[float, float] = (-13.0, -8.0),
    n_scan: int = 200,
    fit_mode: str = "linear",
) -> ScaleScanResult:
    """Scan fit quality across a range of acceleration scales.

    Replaces g† with each trial scale and measures how sharply the
    fit prefers the true g†.

    Parameters
    ----------
    kernel_name : str
        Kernel to test.
    x, y : arrays
        Physical data.
    log_scale_range : tuple
        (log10_lo, log10_hi) range for scale scan.
    n_scan : int
        Number of grid points.
    fit_mode : str
        Fitting mode ("linear", "log", "direct").

    Returns
    -------
    ScaleScanResult
    """
    kernel_fn = KERNEL_REGISTRY[kernel_name]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    n_params_map = {"linear": 2, "log": 2, "direct": 0}
    k_nuisance = n_params_map[fit_mode]

    log_scales = np.linspace(log_scale_range[0], log_scale_range[1], n_scan)
    rms_arr = np.zeros(n_scan)
    aic_arr = np.zeros(n_scan)

    for i, ls in enumerate(log_scales):
        s = 10**ls
        _, _, rms = _fit_kernel_at_scale(kernel_fn, x, y, s, fit_mode)
        rms_arr[i] = rms
        log_like = -0.5 * n * np.log(2 * np.pi * rms**2 + 1e-300) - n / 2
        aic_arr[i] = 2 * k_nuisance - 2 * log_like

    best_idx = int(np.argmin(rms_arr))
    best_log_scale = float(log_scales[best_idx])
    best_rms = float(rms_arr[best_idx])

    # AIC at g†
    gdagger_idx = int(np.argmin(np.abs(log_scales - LOG_G_DAGGER)))
    delta_aic_gd = float(aic_arr[gdagger_idx] - aic_arr[best_idx])

    # Peak sharpness: second derivative of AIC at optimum
    if 1 <= best_idx <= n_scan - 2:
        d2 = ((aic_arr[best_idx + 1] - 2 * aic_arr[best_idx]
               + aic_arr[best_idx - 1])
              / (log_scales[1] - log_scales[0])**2)
        peak_sharpness = float(d2)
    else:
        peak_sharpness = 0.0

    within = abs(best_log_scale - LOG_G_DAGGER) < 0.1

    # ΔAIC at ±0.1 dex from best scale
    delta_aic_neighbors: Dict[str, float] = {}
    for label, offset in [("minus_0p1", -0.1), ("plus_0p1", +0.1)]:
        neighbor_log = best_log_scale + offset
        idx_nb = int(np.argmin(np.abs(log_scales - neighbor_log)))
        delta_aic_neighbors[label] = float(
            aic_arr[idx_nb] - aic_arr[best_idx]
        )
    # Also store the average (symmetric sharpness indicator)
    delta_aic_neighbors["mean_0p1"] = float(
        0.5 * (delta_aic_neighbors["minus_0p1"]
               + delta_aic_neighbors["plus_0p1"])
    )

    # Normalized by N: dataset-size-independent sharpness
    delta_aic_per_dof: Dict[str, float] = {
        k: v / max(n, 1) for k, v in delta_aic_neighbors.items()
    }

    return ScaleScanResult(
        log_scales=log_scales.tolist(),
        rms_values=rms_arr.tolist(),
        aic_values=aic_arr.tolist(),
        best_log_scale=best_log_scale,
        best_rms=best_rms,
        delta_aic_at_gdagger=delta_aic_gd,
        peak_sharpness=peak_sharpness,
        within_0p1_dex=within,
        delta_aic_pm_0p1_dex=delta_aic_neighbors,
        delta_aic_pm_0p1_per_dof=delta_aic_per_dof,
    )


@dataclass
class ShuffleNullResult:
    """Result of shuffle-based null hypothesis test."""
    n_shuffles: int
    observed_rms: float
    null_rms_mean: float
    null_rms_std: float
    null_rms_values: List[float]  # full null RMS distribution
    p_value_rms: float            # fraction of nulls with rms ≤ observed
    observed_best_log_scale: float
    null_best_log_scales: List[float]
    p_value_near_gdagger: float   # fraction of null optima within ±0.1 dex
    p_within_0p1_dex: float = 0.0  # same as above, clearer name for summary
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


def shuffle_null_test(
    kernel_name: str,
    x: NDArray,
    y: NDArray,
    n_shuffles: int = 200,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 100,
    seed: int = 42,
    parallel: bool = False,
    n_workers: int = 4,
    fit_mode: str = "linear",
) -> ShuffleNullResult:
    """Shuffle y relative to x and refit to estimate null distribution.

    Tests whether the observed fit quality and scale preference are
    significant or could arise by chance.

    Parameters
    ----------
    kernel_name : str
        Kernel to test.
    x, y : arrays
        Physical data.
    n_shuffles : int
        Number of random permutations.
    scale_range : tuple
        Search range for acceleration scale.
    n_grid : int
        Grid points for scale search per shuffle.
    seed : int
        Random seed.
    parallel : bool
        If True, use joblib for parallel shuffles.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    ShuffleNullResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Observed fit
    obs_result = fit_kernel(kernel_name, x, y,
                            scale_range=scale_range, n_grid=n_grid,
                            rng_seed=seed, fit_mode=fit_mode)
    obs_rms = obs_result.residual_rms
    obs_log_scale = obs_result.log_scale_best

    def _one_shuffle(shuffle_seed: int) -> Tuple[float, float]:
        rng = np.random.RandomState(shuffle_seed)
        y_shuf = rng.permutation(y)
        r = fit_kernel(kernel_name, x, y_shuf,
                       scale_range=scale_range, n_grid=n_grid,
                       rng_seed=shuffle_seed, fit_mode=fit_mode)
        return r.residual_rms, r.log_scale_best

    seeds = [seed + i + 1 for i in range(n_shuffles)]

    if parallel:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers)(
                delayed(_one_shuffle)(s) for s in seeds
            )
        except ImportError:
            logging.warning("joblib not available; running serially.")
            results = [_one_shuffle(s) for s in seeds]
    else:
        results = [_one_shuffle(s) for s in seeds]

    null_rms = np.array([r[0] for r in results])
    null_log_scales = np.array([r[1] for r in results])

    p_rms = float(np.mean(null_rms <= obs_rms))
    p_near_gd = float(np.mean(np.abs(null_log_scales - LOG_G_DAGGER) < 0.1))

    return ShuffleNullResult(
        n_shuffles=n_shuffles,
        observed_rms=float(obs_rms),
        null_rms_mean=float(np.mean(null_rms)),
        null_rms_std=float(np.std(null_rms)),
        null_rms_values=null_rms.tolist(),
        p_value_rms=p_rms,
        observed_best_log_scale=float(obs_log_scale),
        null_best_log_scales=null_log_scales.tolist(),
        p_value_near_gdagger=p_near_gd,
        p_within_0p1_dex=p_near_gd,
        seed=seed,
    )


@dataclass
class NearbyScaleResult:
    """Comparison of g† against nearby characteristic scales."""
    scales_tested: Dict[str, float]     # name → log10(value)
    aic_at_scale: Dict[str, float]
    rms_at_scale: Dict[str, float]
    best_name: str
    best_log_scale: float
    interpretation_notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


def nearby_scale_comparison(
    kernel_name: str,
    x: NDArray,
    y: NDArray,
    fit_mode: str = "linear",
) -> NearbyScaleResult:
    """Compare fit quality at g† against nearby characteristic scales.

    Tests: g†, a_Λ = c²√(Λ/3), cH₀, cH₀/6 (Verlinde), cH₀/2π,
    and 10× and 0.1× multiples.
    """
    scales = {
        "g†":            G_DAGGER,
        "a_Lambda":      A_LAMBDA,
        "cH0":           A_HUBBLE,
        "cH0/6":         A_HUBBLE / 6.0,
        "cH0/(2pi)":     A_HUBBLE / (2 * np.pi),
        "10*g†":         10 * G_DAGGER,
        "0.1*g†":        0.1 * G_DAGGER,
        "100*g†":        100 * G_DAGGER,
    }

    kernel_fn = KERNEL_REGISTRY[kernel_name]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    aic_map: Dict[str, float] = {}
    rms_map: Dict[str, float] = {}
    log_map: Dict[str, float] = {}

    n_params_map = {"linear": 2, "log": 2, "direct": 0}
    k_nuisance = n_params_map[fit_mode]

    for name, val in scales.items():
        _, _, rms = _fit_kernel_at_scale(kernel_fn, x, y, val, fit_mode)
        log_like = -0.5 * n * np.log(2 * np.pi * rms**2 + 1e-300) - n / 2
        aic = 2 * k_nuisance - 2 * log_like
        aic_map[name] = float(aic)
        rms_map[name] = float(rms)
        log_map[name] = float(np.log10(val))

    best_name = min(aic_map, key=aic_map.get)

    # Interpretation guardrails
    notes: Dict[str, str] = {}
    if "a_Lambda" in aic_map and "g†" in aic_map:
        delta_lambda = aic_map["a_Lambda"] - aic_map["g†"]
        if delta_lambda > 2:
            notes["a_Lambda_caveat"] = (
                f"a_Lambda disfavored by ΔAIC={delta_lambda:.1f} vs g†, "
                "but this tests eta=1 (pure normalization) only. "
                "It proves eta != 1, not that Lambda cannot set the "
                "scale. Fair statement: 'pure a_Lambda normalization "
                "is disfavored unless an order-unity prefactor is "
                "allowed (Verlinde cH0/6 provides eta ~ 0.18).'"
            )
    if "cH0/6" in aic_map and "g†" in aic_map:
        delta_v = aic_map["cH0/6"] - aic_map["g†"]
        notes["cH0_6_note"] = (
            f"Verlinde cH0/6 vs g†: ΔAIC={delta_v:.1f}. "
            f"log(cH0/6)={log_map['cH0/6']:.3f} vs "
            f"log(g†)={LOG_G_DAGGER:.3f} "
            f"(Δ={abs(log_map['cH0/6'] - LOG_G_DAGGER):.3f} dex, "
            f"~1.5% match in linear)."
        )

    return NearbyScaleResult(
        scales_tested=log_map,
        aic_at_scale=aic_map,
        rms_at_scale=rms_map,
        best_name=best_name,
        best_log_scale=log_map[best_name],
        interpretation_notes=notes,
    )


# ============================================================
# §4  DATA LOADERS (for repo-local datasets)
# ============================================================

def load_sparc_rar(
    project_root: Optional[str] = None,
    y_disk: float = 0.5,
    y_bulge: float = 0.7,
    q_max: int = 2,
    inc_range: Tuple[float, float] = (30.0, 85.0),
    min_points: int = 5,
) -> Tuple[NDArray, NDArray]:
    """Load SPARC RAR data (log g_bar, log g_obs) with standard quality cuts.

    Returns
    -------
    (log_gbar, log_gobs) : arrays in m/s² units (log10).
    """
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)

    data_dir = os.path.join(project_root, "data", "sparc")
    table2 = os.path.join(data_dir, "SPARC_table2_rotmods.dat")
    mrt = os.path.join(data_dir, "SPARC_Lelli2016c.mrt")

    if not os.path.exists(table2):
        raise FileNotFoundError(f"SPARC table2 not found: {table2}")

    # Parse table2 (fixed-width)
    galaxies: Dict[str, Dict[str, Any]] = {}
    with open(table2, "r") as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                dist = float(line[12:18].strip())
                rad = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                evobs = float(line[33:38].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue
            if name not in galaxies:
                galaxies[name] = {
                    "R": [], "Vobs": [], "eVobs": [],
                    "Vgas": [], "Vdisk": [], "Vbul": [],
                    "dist": dist,
                }
            g = galaxies[name]
            g["R"].append(rad)
            g["Vobs"].append(vobs)
            g["eVobs"].append(evobs)
            g["Vgas"].append(vgas)
            g["Vdisk"].append(vdisk)
            g["Vbul"].append(vbul)

    for name in galaxies:
        for key in ("R", "Vobs", "eVobs", "Vgas", "Vdisk", "Vbul"):
            galaxies[name][key] = np.array(galaxies[name][key])

    # Parse MRT for quality/inclination
    props: Dict[str, Dict[str, float]] = {}
    if os.path.exists(mrt):
        with open(mrt, "r") as f:
            lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("---") and i > 50:
                data_start = i + 1
                break
        for line in lines[data_start:]:
            if len(line.strip()) < 20:
                continue
            name = line[0:11].strip()
            parts = line[11:].split()
            if len(parts) < 17:
                continue
            try:
                props[name] = {
                    "Inc": float(parts[4]),
                    "Q": int(parts[16]),
                }
            except (ValueError, IndexError):
                continue

    # Quality cuts + RAR computation
    all_log_gbar: List[float] = []
    all_log_gobs: List[float] = []

    for name, g in galaxies.items():
        if name in props:
            if props[name]["Q"] > q_max:
                continue
            if not (inc_range[0] < props[name]["Inc"] < inc_range[1]):
                continue

        R = g["R"]
        Vobs = g["Vobs"]
        Vgas = g["Vgas"]
        Vdisk = g["Vdisk"]
        Vbul = g["Vbul"]

        if len(R) < min_points:
            continue

        # Baryonic velocity squared
        Vbar_sq = (y_disk * Vdisk**2
                   + Vgas * np.abs(Vgas)
                   + y_bulge * Vbul * np.abs(Vbul))

        # Accelerations in m/s²
        r_m = R * KPC_M
        g_bar = np.abs(Vbar_sq) * (1e3)**2 / r_m
        g_obs = (Vobs * 1e3)**2 / r_m

        valid = (g_bar > 0) & (g_obs > 0) & (R > 0)
        if np.sum(valid) < min_points:
            continue

        all_log_gbar.extend(np.log10(g_bar[valid]))
        all_log_gobs.extend(np.log10(g_obs[valid]))

    return np.array(all_log_gbar), np.array(all_log_gobs)


def load_tian2020_cluster_rar(
    project_root: Optional[str] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load Tian+2020 CLASH cluster RAR data.

    Returns
    -------
    (log_gbar, log_gtot, e_log_gbar, e_log_gtot) : arrays
    """
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)

    path = os.path.join(project_root, "data", "cluster_rar",
                        "tian2020_fig2.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tian+2020 data not found: {path}")

    log_gbar_list: List[float] = []
    log_gtot_list: List[float] = []
    e_gbar_list: List[float] = []
    e_gtot_list: List[float] = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            try:
                lg_bar = float(parts[2].strip())
                lg_tot = float(parts[3].strip())
                e_bar = float(parts[4].strip())
                e_tot = float(parts[5].strip())
            except (ValueError, IndexError):
                continue
            log_gbar_list.append(lg_bar)
            log_gtot_list.append(lg_tot)
            e_gbar_list.append(e_bar)
            e_gtot_list.append(e_tot)

    return (np.array(log_gbar_list), np.array(log_gtot_list),
            np.array(e_gbar_list), np.array(e_gtot_list))


# ============================================================
# §5  SYNTHETIC DATA GENERATOR
# ============================================================

def generate_synthetic(
    kernel_name: str = "BE_RAR",
    true_scale: float = G_DAGGER,
    n_points: int = 500,
    x_range: Tuple[float, float] = (1e-13, 1e-9),
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> Tuple[NDArray, NDArray, Dict[str, Any]]:
    """Generate synthetic data from a known kernel + scale.

    Parameters
    ----------
    kernel_name : str
        Kernel to use for generation.
    true_scale : float
        True acceleration scale.
    n_points : int
        Number of data points.
    x_range : tuple
        (lo, hi) for log-uniform x sampling.
    amplitude, offset : float
        y = amplitude * kernel(x, true_scale) + offset + noise.
    noise_sigma : float
        Gaussian noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    x, y : arrays
    truth : dict with generation parameters
    """
    rng = np.random.RandomState(seed)
    kernel_fn = KERNEL_REGISTRY[kernel_name]

    log_x = rng.uniform(np.log10(x_range[0]), np.log10(x_range[1]), n_points)
    x = 10**log_x
    y_clean = amplitude * kernel_fn(x, true_scale) + offset
    noise = rng.normal(0, noise_sigma, n_points)
    y = y_clean + noise

    truth = {
        "kernel_name": kernel_name,
        "true_scale": float(true_scale),
        "true_log_scale": float(np.log10(true_scale)),
        "amplitude": float(amplitude),
        "offset": float(offset),
        "noise_sigma": float(noise_sigma),
        "n_points": n_points,
        "seed": seed,
    }
    return x, y, truth


# ============================================================
# §6  EXPERIMENT RUNNER
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a g† Hunt experiment."""
    tag: str = "pilot"
    seed: int = 42
    kernel_names: List[str] = field(
        default_factory=lambda: list(KERNEL_REGISTRY.keys())
    )
    scale_range: Tuple[float, float] = (1e-13, 1e-8)
    n_grid: int = 200
    n_scan: int = 300
    n_shuffles: int = 200
    n_cv_folds: int = 5
    n_bootstrap: int = 0
    parallel: bool = False
    n_workers: int = 4
    output_base: str = "outputs/gdagger_hunt"
    fit_mode: str = "linear"  # "linear", "log", or "direct"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scale_range"] = list(d["scale_range"])
        return d


def run_experiment(
    x: NDArray,
    y: NDArray,
    config: ExperimentConfig,
    dataset_name: str = "unknown",
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a full g† Hunt experiment: kernel matching + all controls.

    Writes results to a timestamped directory.

    Parameters
    ----------
    x, y : arrays
        Physical data (x = acceleration or equivalent, y = response).
    config : ExperimentConfig
        Run parameters.
    dataset_name : str
        Label for this dataset.
    dataset_meta : dict or None
        Extra metadata to store.

    Returns
    -------
    summary : dict
        Full summary including paths, metrics, verdicts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Create output directory
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_base, f"{ts}_{config.tag}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(run_dir, "logs.txt")
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logger = logging.getLogger("gdagger_hunt")
    logger.info("Starting g† Hunt: dataset=%s, tag=%s", dataset_name,
                config.tag)
    logger.info("N_data=%d, seed=%d", len(x), config.seed)

    # Save params
    params = {
        "config": config.to_dict(),
        "dataset_name": dataset_name,
        "dataset_meta": dataset_meta or {},
        "n_data": len(x),
        "timestamp": ts,
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)

    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "timestamp": ts,
        "n_data": len(x),
    }

    # --- Step 1: Kernel matching ---
    logger.info("Step 1: Kernel matching across %d kernels",
                len(config.kernel_names))
    kernel_results = match_kernels(
        x, y,
        kernel_names=config.kernel_names,
        scale_range=config.scale_range,
        n_grid=config.n_grid,
        n_cv_folds=config.n_cv_folds,
        rng_seed=config.seed,
        fit_mode=config.fit_mode,
    )
    summary["kernel_ranking"] = [r.to_dict() for r in kernel_results]
    if kernel_results:
        best_kernel = kernel_results[0]
        summary["best_kernel"] = best_kernel.to_dict()
        summary["best_kernel_name"] = best_kernel.kernel_name
        summary["best_log_scale"] = best_kernel.log_scale_best
        summary["best_within_0p1_dex"] = (
            abs(best_kernel.log_scale_best - LOG_G_DAGGER) < 0.1
        )
        logger.info("Best kernel: %s at log scale %.3f (Δg†=%.3f dex)",
                     best_kernel.kernel_name,
                     best_kernel.log_scale_best,
                     best_kernel.log_scale_best - LOG_G_DAGGER)

    # --- Step 2: Scale injection scan (for best kernel) ---
    if kernel_results:
        logger.info("Step 2: Scale injection scan")
        scan = scale_injection_scan(
            best_kernel.kernel_name, x, y,
            log_scale_range=(np.log10(config.scale_range[0]),
                             np.log10(config.scale_range[1])),
            n_scan=config.n_scan,
            fit_mode=config.fit_mode,
        )
        summary["scale_scan"] = scan.to_dict()
        logger.info("Scale scan: best=%.3f, ΔAIC(g†)=%.1f, sharp=%.1f",
                     scan.best_log_scale, scan.delta_aic_at_gdagger,
                     scan.peak_sharpness)

        # Plot: AIC vs scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan.log_scales, scan.aic_values, "b-", lw=1.5)
        ax.axvline(LOG_G_DAGGER, color="red", ls="--", lw=2,
                    label=f"g† = {LOG_G_DAGGER:.2f}")
        ax.axvline(scan.best_log_scale, color="green", ls=":",
                    label=f"best = {scan.best_log_scale:.2f}")
        ax.set_xlabel("log₁₀(scale / m s⁻²)")
        ax.set_ylabel("AIC")
        ax.set_title(f"Scale Scan — {best_kernel.kernel_name} "
                      f"on {dataset_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "scale_scan_aic.png"), dpi=150)
        plt.close(fig)

        # Plot: RMS vs scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan.log_scales, scan.rms_values, "b-", lw=1.5)
        ax.axvline(LOG_G_DAGGER, color="red", ls="--", lw=2,
                    label=f"g† = {LOG_G_DAGGER:.2f}")
        ax.axvline(scan.best_log_scale, color="green", ls=":",
                    label=f"best = {scan.best_log_scale:.2f}")
        ax.set_xlabel("log₁₀(scale / m s⁻²)")
        ax.set_ylabel("RMS residual")
        ax.set_title(f"Scale Scan — {best_kernel.kernel_name} "
                      f"on {dataset_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "scale_scan_rms.png"), dpi=150)
        plt.close(fig)

    # --- Step 3: Nearby-scale comparison ---
    if kernel_results:
        logger.info("Step 3: Nearby-scale comparison")
        nearby = nearby_scale_comparison(best_kernel.kernel_name, x, y,
                                         fit_mode=config.fit_mode)
        summary["nearby_scales"] = nearby.to_dict()
        logger.info("Nearby-scale best: %s (log=%.3f)",
                     nearby.best_name, nearby.best_log_scale)

        # Plot: bar chart of AIC at each named scale
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(nearby.aic_at_scale.keys())
        aics = [nearby.aic_at_scale[n] for n in names]
        colors = ["red" if n == "g†" else "steelblue" for n in names]
        bars = ax.bar(range(len(names)), aics, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("AIC")
        ax.set_title(f"Nearby Scale Comparison — {dataset_name}")
        # Highlight best
        best_i = names.index(nearby.best_name)
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "nearby_scale_aic.png"), dpi=150)
        plt.close(fig)

    # --- Step 4: Shuffle null test ---
    if kernel_results and config.n_shuffles > 0:
        logger.info("Step 4: Shuffle null test (%d shuffles)",
                     config.n_shuffles)
        shuf = shuffle_null_test(
            best_kernel.kernel_name, x, y,
            n_shuffles=config.n_shuffles,
            scale_range=config.scale_range,
            n_grid=min(config.n_grid, 100),
            seed=config.seed,
            parallel=config.parallel,
            n_workers=config.n_workers,
            fit_mode=config.fit_mode,
        )
        summary["shuffle_null"] = shuf.to_dict()
        logger.info("Shuffle: p(rms)=%.4f, p(near g†)=%.4f",
                     shuf.p_value_rms, shuf.p_value_near_gdagger)

        # Plot: null distribution of best log-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(shuf.null_best_log_scales, bins=40, alpha=0.7,
                color="gray", label="Shuffle null")
        ax.axvline(shuf.observed_best_log_scale, color="blue", lw=2,
                    label=f"Observed = {shuf.observed_best_log_scale:.2f}")
        ax.axvline(LOG_G_DAGGER, color="red", ls="--", lw=2,
                    label=f"g† = {LOG_G_DAGGER:.2f}")
        ax.set_xlabel("log₁₀(best scale)")
        ax.set_ylabel("Count")
        ax.set_title(f"Shuffle Null — {dataset_name} "
                      f"(p_near_g†={shuf.p_value_near_gdagger:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "shuffle_null_scales.png"),
                    dpi=150)
        plt.close(fig)

        # Plot: null distribution of RMS (using stored values)
        fig, ax = plt.subplots(figsize=(8, 5))
        null_rms = np.array(shuf.null_rms_values)
        ax.hist(null_rms, bins=30, alpha=0.7, color="gray",
                label="Shuffle null")
        ax.axvline(shuf.observed_rms, color="blue", lw=2,
                    label=f"Observed = {shuf.observed_rms:.4f}")
        ax.set_xlabel("RMS residual")
        ax.set_ylabel("Count")
        ax.set_title(f"RMS Null Distribution — {dataset_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "shuffle_null_rms.png"), dpi=150)
        plt.close(fig)

    # --- Step 5: Kernel comparison plot ---
    if kernel_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        names_k = [r.kernel_name for r in kernel_results]
        aics_k = [r.aic for r in kernel_results]
        colors_k = ["red" if r.kernel_name == best_kernel.kernel_name
                     else "steelblue" for r in kernel_results]
        ax.barh(range(len(names_k)), aics_k, color=colors_k)
        ax.set_yticks(range(len(names_k)))
        ax.set_yticklabels(names_k)
        ax.set_xlabel("AIC (lower is better)")
        ax.set_title(f"Kernel Comparison — {dataset_name}")
        # Add scale annotation
        for i, r in enumerate(kernel_results):
            ax.text(r.aic + 0.5, i,
                    f"log s={r.log_scale_best:.2f}",
                    va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "kernel_comparison.png"), dpi=150)
        plt.close(fig)

    # --- Step 6: Data overview plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.log10(x + 1e-300), y, s=3, alpha=0.3, c="navy")
    if kernel_results:
        x_plot = np.logspace(np.log10(x.min() + 1e-300),
                             np.log10(x.max()), 500)
        k_fn = KERNEL_REGISTRY[best_kernel.kernel_name]
        k_raw = k_fn(x_plot, best_kernel.scale_best)
        if config.fit_mode == "direct":
            y_plot = k_raw
        elif config.fit_mode == "log":
            y_plot = 10**(best_kernel.amplitude * np.log10(k_raw + 1e-300) + best_kernel.offset)
        else:
            y_plot = best_kernel.amplitude * k_raw + best_kernel.offset
        ax.plot(np.log10(x_plot), y_plot, "r-", lw=2,
                label=f"{best_kernel.kernel_name} (s={best_kernel.log_scale_best:.2f})")
        ax.legend()
    ax.set_xlabel("log₁₀(x)")
    ax.set_ylabel("y")
    ax.set_title(f"Data + Best Fit — {dataset_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "data_overview.png"), dpi=150)
    plt.close(fig)

    # --- Verdicts ---
    verdicts: Dict[str, str] = {}
    if kernel_results:
        bk = kernel_results[0]
        delta = abs(bk.log_scale_best - LOG_G_DAGGER)
        if delta < 0.05:
            verdicts["scale_recovery"] = "STRONG_HIT"
        elif delta < 0.1:
            verdicts["scale_recovery"] = "HIT"
        elif delta < 0.3:
            verdicts["scale_recovery"] = "MARGINAL"
        else:
            verdicts["scale_recovery"] = "MISS"

        if bk.kernel_name in ("BE_RAR", "BE_occupation", "BE_cousin"):
            verdicts["kernel_family"] = "BE_FAMILY"
        else:
            verdicts["kernel_family"] = "NON_BE"

        if "shuffle_null" in summary:
            p = summary["shuffle_null"]["p_value_near_gdagger"]
            if p < 0.05:
                verdicts["shuffle_control"] = "SIGNIFICANT"
            elif p < 0.1:
                verdicts["shuffle_control"] = "MARGINAL"
            else:
                verdicts["shuffle_control"] = "NOT_SIGNIFICANT"

        if "scale_scan" in summary:
            sharp = summary["scale_scan"]["peak_sharpness"]
            if sharp > 50:
                verdicts["peak_sharpness"] = "SHARP"
            elif sharp > 10:
                verdicts["peak_sharpness"] = "MODERATE"
            else:
                verdicts["peak_sharpness"] = "BROAD"

    summary["verdicts"] = verdicts
    logger.info("Verdicts: %s", verdicts)

    # --- Write outputs ---
    # Summary JSON
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Metrics Parquet (try pandas, fall back to CSV)
    try:
        import pandas as pd
        metrics_df = pd.DataFrame([r.to_dict() for r in kernel_results])
        metrics_df.to_parquet(os.path.join(run_dir, "metrics.parquet"))
    except Exception:
        import csv
        if kernel_results:
            keys = kernel_results[0].to_dict().keys()
            with open(os.path.join(run_dir, "metrics.csv"), "w",
                      newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in kernel_results:
                    writer.writerow(r.to_dict())

    logger.info("Outputs written to %s", run_dir)
    summary["output_dir"] = run_dir
    return summary


# ============================================================
# §6b  THREE-PANEL SUMMARY FIGURE
# ============================================================

# Named reference scales for vertical lines
_REFERENCE_SCALES: Dict[str, Tuple[float, str, str]] = {
    # name → (log10_value, color, linestyle)
    "g†":     (LOG_G_DAGGER,                   "red",       "-"),
    "cH₀/6":  (np.log10(A_HUBBLE / 6.0),       "darkorange", "--"),
    "a_Λ":    (np.log10(A_LAMBDA),              "purple",    ":"),
}


def plot_three_panel_summary(
    summaries: Dict[str, Dict[str, Any]],
    sparc_key: str = "sparc_rar",
    output_path: Optional[str] = None,
    dpi: int = 200,
) -> str:
    """Create 3-panel debate-ending figure.

    Panel 1: AIC vs log-scale (from sparc_key) with reference lines.
    Panel 2: Null distribution of best scales (histogram) with same lines.
    Panel 3: ΔAIC/N at ±0.1 dex bar chart across all datasets.

    Parameters
    ----------
    summaries : dict
        {dataset_label: summary_dict} from run_experiment().
    sparc_key : str
        Key into summaries for the primary RAR dataset (Panel 1 & 2).
    output_path : str or None
        Where to save. If None, saves to first dataset's output_dir.
    dpi : int
        Figure resolution.

    Returns
    -------
    str : path to saved figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Panel 1: AIC vs log-scale (SPARC) ----
    ax1 = axes[0]
    sparc = summaries.get(sparc_key, {})
    scan = sparc.get("scale_scan", {})

    if scan:
        log_s = np.array(scan["log_scales"])
        aic = np.array(scan["aic_values"])
        # Normalize to ΔAIC from best
        aic_best = np.min(aic)
        ax1.plot(log_s, aic - aic_best, "k-", lw=1.2, alpha=0.9)

        # Best scale
        best_ls = scan["best_log_scale"]
        ax1.axvline(best_ls, color="green", ls="-.", lw=1.5, alpha=0.8,
                     label=f"best = {best_ls:.2f}")

        # Reference scales
        for name, (val, color, ls) in _REFERENCE_SCALES.items():
            ax1.axvline(val, color=color, ls=ls, lw=1.8,
                         label=f"{name} = {val:.2f}")

        ax1.set_xlabel("log₁₀(scale / m s⁻²)", fontsize=10)
        ax1.set_ylabel("ΔAIC (from best)", fontsize=10)
        ax1.set_title("(a) Scale preference — SPARC RAR", fontsize=11,
                        fontweight="bold")
        ax1.legend(fontsize=7, loc="upper left")

        # Zoom to interesting range
        ax1.set_xlim(best_ls - 1.5, best_ls + 1.5)
        y_max = float(np.max(aic[
            (log_s > best_ls - 1.5) & (log_s < best_ls + 1.5)
        ] - aic_best))
        ax1.set_ylim(-5, min(y_max * 1.1, 2000))
    else:
        ax1.text(0.5, 0.5, f"No scale_scan in '{sparc_key}'",
                 ha="center", va="center", transform=ax1.transAxes)

    # ---- Panel 2: Null distribution of best scales ----
    ax2 = axes[1]
    shuf = sparc.get("shuffle_null", {})

    if shuf:
        null_scales = np.array(shuf["null_best_log_scales"])
        obs_scale = shuf["observed_best_log_scale"]
        p_near = shuf.get("p_within_0p1_dex",
                           shuf.get("p_value_near_gdagger", -1))

        ax2.hist(null_scales, bins=min(40, max(10, len(null_scales) // 5)),
                 alpha=0.65, color="gray", edgecolor="dimgray",
                 label="Shuffle null")
        ax2.axvline(obs_scale, color="green", ls="-.", lw=1.8,
                     label=f"observed = {obs_scale:.2f}")

        # Same reference scales
        for name, (val, color, ls) in _REFERENCE_SCALES.items():
            ax2.axvline(val, color=color, ls=ls, lw=1.8,
                         label=f"{name} = {val:.2f}")

        # ±0.1 dex band around g†
        ax2.axvspan(LOG_G_DAGGER - 0.1, LOG_G_DAGGER + 0.1,
                     alpha=0.08, color="red")

        ax2.set_xlabel("log₁₀(best scale from shuffle)", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title(f"(b) Shuffle null — p(near g†) = {p_near:.3f}",
                        fontsize=11, fontweight="bold")
        ax2.legend(fontsize=7, loc="upper left")
    else:
        ax2.text(0.5, 0.5, "No shuffle_null data", ha="center",
                 va="center", transform=ax2.transAxes)

    # ---- Panel 3: ΔAIC/N at ±0.1 dex (bar chart across datasets) ----
    ax3 = axes[2]
    labels: List[str] = []
    minus_vals: List[float] = []
    plus_vals: List[float] = []
    mean_vals: List[float] = []

    for dset_label, s in summaries.items():
        ss = s.get("scale_scan", {})
        per_dof = ss.get("delta_aic_pm_0p1_per_dof", {})
        if per_dof:
            labels.append(dset_label.replace("_", "\n"))
            minus_vals.append(per_dof.get("minus_0p1", 0))
            plus_vals.append(per_dof.get("plus_0p1", 0))
            mean_vals.append(per_dof.get("mean_0p1", 0))

    if labels:
        x_pos = np.arange(len(labels))
        width = 0.25
        ax3.bar(x_pos - width, minus_vals, width, label="−0.1 dex",
                color="steelblue", alpha=0.85)
        ax3.bar(x_pos, plus_vals, width, label="+0.1 dex",
                color="darkorange", alpha=0.85)
        ax3.bar(x_pos + width, mean_vals, width, label="mean",
                color="seagreen", alpha=0.85)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylabel("ΔAIC / N  at ±0.1 dex", fontsize=10)
        ax3.set_title("(c) Scale identifiability", fontsize=11,
                        fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.axhline(0, color="gray", lw=0.5)
    else:
        ax3.text(0.5, 0.5, "No ΔAIC/N data", ha="center",
                 va="center", transform=ax3.transAxes)

    fig.tight_layout()

    if output_path is None:
        # Find first available output_dir
        for s in summaries.values():
            od = s.get("output_dir")
            if od:
                output_path = os.path.join(
                    os.path.dirname(od), "three_panel_summary.png")
                break
        if output_path is None:
            output_path = "three_panel_summary.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# §7  CLI ENTRYPOINT
# ============================================================

def _load_config_from_yaml(path: str) -> ExperimentConfig:
    """Load ExperimentConfig from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML config. "
                          "Install: pip install pyyaml")

    with open(path, "r") as f:
        d = yaml.safe_load(f)

    cfg = d.get("gdagger_hunt", d)
    return ExperimentConfig(
        tag=cfg.get("tag", "cli"),
        seed=cfg.get("seed", 42),
        kernel_names=cfg.get("kernel_names", list(KERNEL_REGISTRY.keys())),
        scale_range=tuple(cfg.get("scale_range", [1e-13, 1e-8])),
        n_grid=cfg.get("n_grid", 200),
        n_scan=cfg.get("n_scan", 300),
        n_shuffles=cfg.get("n_shuffles", 200),
        n_cv_folds=cfg.get("n_cv_folds", 5),
        n_bootstrap=cfg.get("n_bootstrap", 0),
        parallel=cfg.get("parallel", False),
        n_workers=cfg.get("n_workers", 4),
        output_base=cfg.get("output_base", "outputs/gdagger_hunt"),
    )


def main() -> None:
    """CLI entrypoint for g† Hunt experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="g† Hunt: Systematic search for BEC acceleration scale"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")
    parser.add_argument("--dataset", type=str, default="sparc_rar",
                        choices=["sparc_rar", "tian2020_cluster",
                                 "synthetic"],
                        help="Dataset to analyze")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag (overrides config)")
    parser.add_argument("--n-shuffles", type=int, default=None,
                        help="Number of shuffles (overrides config)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel shuffles")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-base", type=str, default=None)

    args = parser.parse_args()

    # Load config
    if args.config:
        config = _load_config_from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # Override from CLI
    if args.tag:
        config.tag = args.tag
    if args.n_shuffles is not None:
        config.n_shuffles = args.n_shuffles
    if args.parallel:
        config.parallel = True
    if args.n_workers:
        config.n_workers = args.n_workers
    if args.seed is not None:
        config.seed = args.seed
    if args.output_base:
        config.output_base = args.output_base

    # Load data
    if args.dataset == "sparc_rar":
        log_gbar, log_gobs = load_sparc_rar()
        x = 10**log_gbar  # linear acceleration
        y = 10**log_gobs / 10**log_gbar  # g_obs / g_bar
        dataset_name = "SPARC_RAR"
        meta = {"n_points": len(x), "source": "SPARC table2"}
    elif args.dataset == "tian2020_cluster":
        lg, lt, eg, et = load_tian2020_cluster_rar()
        x = 10**lg
        y = 10**lt / 10**lg
        dataset_name = "Tian2020_Cluster"
        meta = {"n_points": len(x), "source": "Tian+2020 CLASH"}
    elif args.dataset == "synthetic":
        x, y, truth = generate_synthetic(seed=config.seed)
        dataset_name = "Synthetic_BE_RAR"
        meta = truth
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    config.tag = config.tag or args.dataset
    summary = run_experiment(x, y, config,
                             dataset_name=dataset_name,
                             dataset_meta=meta)

    print(f"\n{'='*60}")
    print(f"g† Hunt complete: {dataset_name}")
    print(f"Output: {summary['output_dir']}")
    if "verdicts" in summary:
        for k, v in summary["verdicts"].items():
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
