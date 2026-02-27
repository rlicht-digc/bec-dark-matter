#!/usr/bin/env python3
"""
test_gdagger_hunt_refereeproof.py — Adversarial validation suite for g† Hunt.

Runs a comprehensive battery of tests designed to withstand hostile referee
scrutiny. All results written to a single timestamped "refereeproof" folder.

Validation Suites:
  A) Shuffle nulls (global + within-bin + within-galaxy circular shift)
  B) Grouped cross-validation (GroupKFold by galaxy)
  C) Cut/sample sensitivity (inclination, quality, min-points)
  D) Grid/optimization invariance (n_grid = 100, 300, 1000)
  E) Negative controls / break tests (noise, warp, galaxy swap)
  F) Nearby-scale comparison with matched DoF
  G) Non-RAR pilot (cluster, explicit null example)

Usage:
  python3 analysis/tests/test_gdagger_hunt_refereeproof.py
  python3 analysis/tests/test_gdagger_hunt_refereeproof.py --n-shuffles 1000
  python3 analysis/tests/test_gdagger_hunt_refereeproof.py --quick  # fast mode

Russell Licht — BEC Dark Matter Project, Feb 2026
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from gdagger_hunt import (
    A_HUBBLE,
    A_LAMBDA,
    ExperimentConfig,
    G_DAGGER,
    KERNEL_REGISTRY,
    KPC_M,
    LOG_G_DAGGER,
    _fit_kernel_at_scale,
    _jsonable,
    fit_kernel,
    load_sparc_rar,
    load_tian2020_cluster_rar,
    match_kernels,
    nearby_scale_comparison,
    plot_three_panel_summary,
    run_experiment,
    scale_injection_scan,
    shuffle_null_test,
)


# ============================================================
# EXTENDED SPARC LOADER (returns galaxy IDs)
# ============================================================

def load_sparc_rar_with_galaxies(
    project_root: Optional[str] = None,
    y_disk: float = 0.5,
    y_bulge: float = 0.7,
    q_max: int = 2,
    inc_range: Tuple[float, float] = (30.0, 85.0),
    min_points: int = 5,
) -> Tuple[NDArray, NDArray, NDArray, List[str]]:
    """Load SPARC RAR data with galaxy identifiers for GroupKFold.

    Returns
    -------
    (log_gbar, log_gobs, galaxy_ids_per_point, galaxy_names) :
        galaxy_ids_per_point : int array, same length as log_gbar,
            mapping each point to its galaxy index.
        galaxy_names : list of unique galaxy names.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    data_dir = os.path.join(project_root, "data", "sparc")
    table2 = os.path.join(data_dir, "SPARC_table2_rotmods.dat")
    mrt = os.path.join(data_dir, "SPARC_Lelli2016c.mrt")

    # Parse table2 (fixed-width)
    galaxies: Dict[str, Dict[str, Any]] = {}
    galaxy_order: List[str] = []
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
                galaxy_order.append(name)
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

    # Quality cuts + RAR computation with galaxy tracking
    all_log_gbar: List[float] = []
    all_log_gobs: List[float] = []
    all_gal_idx: List[int] = []
    kept_galaxies: List[str] = []
    gal_counter = 0

    for name in galaxy_order:
        g = galaxies[name]
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

        Vbar_sq = (y_disk * Vdisk**2
                   + Vgas * np.abs(Vgas)
                   + y_bulge * Vbul * np.abs(Vbul))

        r_m = R * KPC_M
        g_bar = np.abs(Vbar_sq) * (1e3)**2 / r_m
        g_obs = (Vobs * 1e3)**2 / r_m

        valid = (g_bar > 0) & (g_obs > 0) & (R > 0)
        if np.sum(valid) < min_points:
            continue

        n_valid = int(np.sum(valid))
        all_log_gbar.extend(np.log10(g_bar[valid]))
        all_log_gobs.extend(np.log10(g_obs[valid]))
        all_gal_idx.extend([gal_counter] * n_valid)
        kept_galaxies.append(name)
        gal_counter += 1

    return (np.array(all_log_gbar), np.array(all_log_gobs),
            np.array(all_gal_idx, dtype=int), kept_galaxies)


# ============================================================
# HELPER: Clopper-Pearson upper bound for p=0
# ============================================================

def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper confidence bound for binomial proportion."""
    if k == 0:
        return 1.0 - alpha**(1.0 / n)
    return float(binom.ppf(1 - alpha, n, 1) / n)  # fallback


# ============================================================
# HELPER: Dual-window proximity
# ============================================================

def compute_dual_window_proximity(
    null_best_log_scales: List[float],
    target: float = LOG_G_DAGGER,
    windows: Tuple[float, ...] = (0.05, 0.10),
) -> Dict[str, Any]:
    """Compute proximity p-values at multiple dex windows.

    Returns dict with keys like "p_within_0p05_dex", "n_hits_0p05", etc.
    """
    arr = np.array(null_best_log_scales)
    n = len(arr)
    result: Dict[str, Any] = {}
    for w in windows:
        wkey = f"{w:.2f}".replace(".", "p")
        n_hits = int(np.sum(np.abs(arr - target) < w))
        p = float(n_hits / max(n, 1))
        result[f"p_within_{wkey}_dex"] = p
        result[f"n_hits_{wkey}"] = n_hits
        if n_hits == 0 and n > 0:
            result[f"p_upper_95_{wkey}"] = float(
                clopper_pearson_upper(0, n))
    return result


# ============================================================
# SUITE A: SHUFFLE NULLS (global + within-bin + within-galaxy)
# ============================================================

def run_suite_a_shuffle_nulls(
    x: NDArray, y: NDArray,
    gal_ids: NDArray,
    kernel_name: str,
    n_shuffles: int = 500,
    seed: int = 42,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 200,
    fit_mode: str = "direct",
) -> Dict[str, Any]:
    """Run shuffle nulls + structure-preserving control + residual-permutation null.

    A1: Global shuffle of y relative to x (destructive null)
    A2: Within-bin shuffle — STRUCTURE-PRESERVING CONTROL (not a null test;
        preserves p(y|x) by construction, so p≈1 is expected)
    A2b: Residual-permutation bin-aware null (destructive): fit best kernel,
         compute residuals, permute residuals within bins, reconstruct y
    A3: Within-galaxy circular shift (destructive null)
    """
    results: Dict[str, Any] = {}
    log_x = np.log10(x + 1e-300)

    # --- A1: Global shuffle ---
    print("    A1: Global shuffle null...")
    t0 = time.time()
    shuf_global = shuffle_null_test(
        kernel_name, x, y,
        n_shuffles=n_shuffles,
        scale_range=scale_range,
        n_grid=min(n_grid, 100),
        seed=seed,
        fit_mode=fit_mode,
    )
    results["A1_global"] = shuf_global.to_dict()
    results["A1_global"]["runtime_s"] = time.time() - t0
    p_g = shuf_global.p_within_0p1_dex
    n_hits_g = int(np.sum(
        np.abs(np.array(shuf_global.null_best_log_scales) - LOG_G_DAGGER) < 0.1
    ))
    results["A1_global"]["n_hits"] = n_hits_g
    results["A1_global"]["n_shuffles"] = n_shuffles
    dw_g = compute_dual_window_proximity(shuf_global.null_best_log_scales)
    results["A1_global"].update(dw_g)
    if n_hits_g == 0:
        results["A1_global"]["p_upper_95"] = float(
            clopper_pearson_upper(0, n_shuffles))
    print(f"      ±0.10 dex: p={dw_g['p_within_0p10_dex']:.4f} "
          f"({dw_g['n_hits_0p10']}/{n_shuffles})")
    print(f"      ±0.05 dex: p={dw_g['p_within_0p05_dex']:.4f} "
          f"({dw_g['n_hits_0p05']}/{n_shuffles})")
    if n_hits_g == 0:
        print(f"      95% Clopper-Pearson UB (±0.10): "
              f"{results['A1_global']['p_upper_95']:.4f}")

    # --- A2: Within-bin shuffle (STRUCTURE-PRESERVING CONTROL) ---
    print("    A2: Within-bin shuffle (structure-preserving control)...")
    print("         [preserves p(y|x) by construction; p≈1 expected]")
    t0 = time.time()
    # Define coarse x-bins (0.5 dex wide)
    bin_edges = np.arange(log_x.min() - 0.01, log_x.max() + 0.51, 0.5)
    bin_idx = np.digitize(log_x, bin_edges)

    null_best_scales_bin: List[float] = []
    null_rms_bin: List[float] = []
    rng = np.random.RandomState(seed + 10000)

    for i_shuf in range(n_shuffles):
        y_shuf = y.copy()
        for b in np.unique(bin_idx):
            mask = bin_idx == b
            y_shuf[mask] = rng.permutation(y_shuf[mask])
        r = fit_kernel(kernel_name, x, y_shuf,
                       scale_range=scale_range, n_grid=min(n_grid, 80),
                       rng_seed=seed + 10000 + i_shuf,
                       fit_mode=fit_mode)
        null_best_scales_bin.append(r.log_scale_best)
        null_rms_bin.append(r.residual_rms)

    null_scales_arr = np.array(null_best_scales_bin)
    n_hits_bin = int(np.sum(np.abs(null_scales_arr - LOG_G_DAGGER) < 0.1))
    p_bin = float(n_hits_bin / n_shuffles)
    dw_bin = compute_dual_window_proximity(null_best_scales_bin)
    results["A2_within_bin"] = {
        "type": "structure_preserving_control",
        "note": ("Within-bin shuffle preserves the conditional distribution "
                 "p(y|x) by construction. It is NOT a null test; p≈1 is the "
                 "expected outcome. This control confirms that the scale "
                 "preference resides in the global x-y correlation, not in "
                 "bin-local structure."),
        "n_shuffles": n_shuffles,
        "null_best_log_scales": null_best_scales_bin,
        "null_rms_values": null_rms_bin,
        "p_within_0p1_dex": p_bin,
        "n_hits": n_hits_bin,
        "bin_width_dex": 0.5,
        "n_bins": len(np.unique(bin_idx)),
        "runtime_s": time.time() - t0,
        **dw_bin,
    }
    print(f"      ±0.10 dex: p={dw_bin['p_within_0p10_dex']:.4f} "
          f"({dw_bin['n_hits_0p10']}/{n_shuffles}) [expected ≈1.0]")
    print(f"      ±0.05 dex: p={dw_bin['p_within_0p05_dex']:.4f} "
          f"({dw_bin['n_hits_0p05']}/{n_shuffles})")

    # --- A2b: Block-permute bins null (DESTRUCTIVE) ---
    print("    A2b: Block-permute bins null (destructive)...")
    t0 = time.time()
    # Shuffle entire bin assignments: for each realization, randomly
    # reassign which bin's y-values go with which bin's x-values.
    # This preserves within-bin y distributions but breaks the
    # cross-bin monotonic relationship that encodes the scale.
    unique_bins = np.unique(bin_idx)
    n_unique_bins = len(unique_bins)

    null_best_scales_block: List[float] = []
    null_rms_block: List[float] = []
    rng2b = np.random.RandomState(seed + 15000)

    for i_shuf in range(n_shuffles):
        # Build block-permuted y: shuffle which bin's y go where
        y_block = y.copy()
        perm_bins = rng2b.permutation(unique_bins)
        bin_map = dict(zip(unique_bins, perm_bins))
        for b_orig, b_dest in bin_map.items():
            if b_orig == b_dest:
                continue
            mask_orig = bin_idx == b_orig
            mask_dest = bin_idx == b_dest
            n_orig = int(np.sum(mask_orig))
            n_dest = int(np.sum(mask_dest))
            n_use = min(n_orig, n_dest)
            if n_use == 0:
                continue
            orig_idx = np.where(mask_orig)[0][:n_use]
            dest_idx = np.where(mask_dest)[0][:n_use]
            y_block[orig_idx] = y[dest_idx]
        r = fit_kernel(kernel_name, x, y_block,
                       scale_range=scale_range, n_grid=min(n_grid, 80),
                       rng_seed=seed + 15000 + i_shuf,
                       fit_mode=fit_mode)
        null_best_scales_block.append(r.log_scale_best)
        null_rms_block.append(r.residual_rms)

    null_scales_block_arr = np.array(null_best_scales_block)
    n_hits_block = int(np.sum(
        np.abs(null_scales_block_arr - LOG_G_DAGGER) < 0.1))
    p_block = float(n_hits_block / n_shuffles)
    dw_block = compute_dual_window_proximity(null_best_scales_block)
    results["A2b_block_permute_bins"] = {
        "type": "destructive_bin_aware_null",
        "note": ("Block-permute bins null: randomly reassign which bin's "
                 "y-values go with which bin's x-values. Preserves the "
                 "within-bin marginal y distribution but breaks the cross-bin "
                 "monotonic relationship that encodes the characteristic "
                 "scale. Intermediate proximity rates are expected for a "
                 "partially structure-preserving but globally destructive "
                 "permutation."),
        "n_shuffles": n_shuffles,
        "null_best_log_scales": null_best_scales_block,
        "null_rms_values": null_rms_block,
        "p_within_0p1_dex": p_block,
        "n_hits": n_hits_block,
        "bin_width_dex": 0.5,
        "n_bins": n_unique_bins,
        "runtime_s": time.time() - t0,
        **dw_block,
    }
    if n_hits_block == 0:
        results["A2b_block_permute_bins"]["p_upper_95"] = float(
            clopper_pearson_upper(0, n_shuffles))
    print(f"      ±0.10 dex: p={dw_block['p_within_0p10_dex']:.4f} "
          f"({dw_block['n_hits_0p10']}/{n_shuffles})")
    print(f"      ±0.05 dex: p={dw_block['p_within_0p05_dex']:.4f} "
          f"({dw_block['n_hits_0p05']}/{n_shuffles})")

    # --- A3: Within-galaxy circular shift ---
    print("    A3: Within-galaxy circular shift null...")
    t0 = time.time()
    unique_gals = np.unique(gal_ids)
    null_best_scales_gal: List[float] = []
    null_rms_gal: List[float] = []
    rng3 = np.random.RandomState(seed + 20000)

    for i_shuf in range(n_shuffles):
        y_shuf = y.copy()
        for gid in unique_gals:
            mask = gal_ids == gid
            n_pts = int(np.sum(mask))
            if n_pts < 2:
                continue
            shift = rng3.randint(1, n_pts)  # non-zero circular shift
            indices = np.where(mask)[0]
            y_shuf[indices] = np.roll(y[indices], shift)
        r = fit_kernel(kernel_name, x, y_shuf,
                       scale_range=scale_range, n_grid=min(n_grid, 80),
                       rng_seed=seed + 20000 + i_shuf,
                       fit_mode=fit_mode)
        null_best_scales_gal.append(r.log_scale_best)
        null_rms_gal.append(r.residual_rms)

    null_scales_gal_arr = np.array(null_best_scales_gal)
    n_hits_gal = int(np.sum(np.abs(null_scales_gal_arr - LOG_G_DAGGER) < 0.1))
    p_gal = float(n_hits_gal / n_shuffles)
    dw_gal = compute_dual_window_proximity(null_best_scales_gal)
    results["A3_within_galaxy"] = {
        "n_shuffles": n_shuffles,
        "null_best_log_scales": null_best_scales_gal,
        "null_rms_values": null_rms_gal,
        "p_within_0p1_dex": p_gal,
        "n_hits": n_hits_gal,
        "n_galaxies": len(unique_gals),
        "runtime_s": time.time() - t0,
        **dw_gal,
    }
    if n_hits_gal == 0:
        results["A3_within_galaxy"]["p_upper_95"] = float(
            clopper_pearson_upper(0, n_shuffles))
    print(f"      ±0.10 dex: p={dw_gal['p_within_0p10_dex']:.4f} "
          f"({dw_gal['n_hits_0p10']}/{n_shuffles})")
    print(f"      ±0.05 dex: p={dw_gal['p_within_0p05_dex']:.4f} "
          f"({dw_gal['n_hits_0p05']}/{n_shuffles})")
    if n_hits_gal == 0:
        print(f"      95% Clopper-Pearson UB (±0.10): "
              f"{results['A3_within_galaxy']['p_upper_95']:.4f}")

    return results


# ============================================================
# SUITE B: GROUPED CROSS-VALIDATION
# ============================================================

def run_suite_b_grouped_cv(
    x: NDArray, y: NDArray,
    gal_ids: NDArray,
    kernel_name: str,
    n_folds: int = 5,
    seed: int = 42,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 200,
    fit_mode: str = "direct",
) -> Dict[str, Any]:
    """GroupKFold cross-validation where group = galaxy."""
    unique_gals = np.unique(gal_ids)
    n_gals = len(unique_gals)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_gals)
    fold_assignment = np.zeros(n_gals, dtype=int)
    for i, gal_perm_idx in enumerate(perm):
        fold_assignment[gal_perm_idx] = i % n_folds

    fold_results: List[Dict[str, Any]] = []
    best_kernels: List[str] = []
    best_scales: List[float] = []
    oos_losses: List[float] = []

    for fold in range(n_folds):
        # Build train/test masks
        test_gals = set(unique_gals[fold_assignment == fold])
        test_mask = np.array([gid in test_gals for gid in gal_ids])
        train_mask = ~test_mask

        if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
            continue

        x_tr, y_tr = x[train_mask], y[train_mask]
        x_te, y_te = x[test_mask], y[test_mask]

        # Fit on train
        results_fold = match_kernels(
            x_tr, y_tr,
            scale_range=scale_range, n_grid=n_grid,
            n_cv_folds=3, rng_seed=seed + fold,
            fit_mode=fit_mode,
        )
        if not results_fold:
            continue

        best = results_fold[0]
        best_kernels.append(best.kernel_name)
        best_scales.append(best.log_scale_best)

        # Evaluate on test
        kernel_fn = KERNEL_REGISTRY[best.kernel_name]
        k_te = kernel_fn(x_te, best.scale_best)
        if fit_mode == "direct":
            pred = k_te
        elif fit_mode == "log":
            pred = 10**(best.amplitude * np.log10(k_te + 1e-300) + best.offset)
        else:
            pred = best.amplitude * k_te + best.offset
        oos_rms = float(np.sqrt(np.mean((y_te - pred)**2)))
        oos_losses.append(oos_rms)

        fold_results.append({
            "fold": fold,
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
            "n_train_galaxies": int(n_gals - len(test_gals)),
            "n_test_galaxies": len(test_gals),
            "best_kernel": best.kernel_name,
            "best_log_scale": best.log_scale_best,
            "train_rms": best.residual_rms,
            "test_rms": oos_rms,
        })

    # Summary statistics
    from collections import Counter
    kernel_counts = Counter(best_kernels)

    result = {
        "n_folds": n_folds,
        "n_galaxies": n_gals,
        "fold_details": fold_results,
        "best_kernel_frequency": dict(kernel_counts),
        "best_kernel_unanimous": len(kernel_counts) == 1,
        "best_log_scale_values": best_scales,
        "best_log_scale_mean": float(np.mean(best_scales)),
        "best_log_scale_std": float(np.std(best_scales)),
        "best_log_scale_range": float(np.ptp(best_scales)),
        "delta_from_gdagger_mean": float(
            np.mean(np.abs(np.array(best_scales) - LOG_G_DAGGER))),
        "oos_rms_values": oos_losses,
        "oos_rms_mean": float(np.mean(oos_losses)),
        "oos_rms_std": float(np.std(oos_losses)),
    }
    return result


# ============================================================
# SUITE C: CUT / SAMPLE SENSITIVITY
# ============================================================

def run_suite_c_cut_sensitivity(
    kernel_name: str,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    n_grid: int = 200,
    fit_mode: str = "direct",
    seed: int = 42,
) -> Dict[str, Any]:
    """Re-run SPARC with stricter cuts and report stability."""
    variants = {
        "baseline": {"q_max": 2, "inc_range": (30.0, 85.0), "min_points": 5},
        "strict_inc": {"q_max": 2, "inc_range": (40.0, 80.0), "min_points": 5},
        "strict_quality": {"q_max": 1, "inc_range": (30.0, 85.0), "min_points": 5},
        "strict_npts": {"q_max": 2, "inc_range": (30.0, 85.0), "min_points": 10},
    }

    results: Dict[str, Any] = {}
    baseline_scale = None

    for name, cuts in variants.items():
        log_gbar, log_gobs = load_sparc_rar(
            project_root=PROJECT_ROOT,
            q_max=cuts["q_max"],
            inc_range=cuts["inc_range"],
            min_points=cuts["min_points"],
        )
        x = 10**log_gbar
        y = 10**log_gobs / 10**log_gbar

        r = fit_kernel(kernel_name, x, y,
                       scale_range=scale_range, n_grid=n_grid,
                       rng_seed=seed, fit_mode=fit_mode)

        scan = scale_injection_scan(
            kernel_name, x, y,
            log_scale_range=(np.log10(scale_range[0]),
                             np.log10(scale_range[1])),
            n_scan=200, fit_mode=fit_mode,
        )

        if name == "baseline":
            baseline_scale = r.log_scale_best

        results[name] = {
            "cuts": cuts,
            "n_data": len(x),
            "best_kernel": r.kernel_name,
            "best_log_scale": r.log_scale_best,
            "delta_from_gdagger": abs(r.log_scale_best - LOG_G_DAGGER),
            "delta_from_baseline": (
                abs(r.log_scale_best - baseline_scale)
                if baseline_scale is not None else 0.0
            ),
            "aic": r.aic,
            "rms": r.residual_rms,
            "peak_sharpness": scan.peak_sharpness,
            "delta_aic_pm_0p1": scan.delta_aic_pm_0p1_dex,
        }

    # Max shift across variants
    scales = [v["best_log_scale"] for v in results.values()]
    results["max_delta_log_scale"] = float(np.ptp(scales))
    results["all_within_0p05_dex"] = float(np.ptp(scales)) < 0.05

    return results


# ============================================================
# SUITE D: GRID / OPTIMIZATION INVARIANCE
# ============================================================

def run_suite_d_grid_invariance(
    x: NDArray, y: NDArray,
    kernel_name: str,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    fit_mode: str = "direct",
    seed: int = 42,
) -> Dict[str, Any]:
    """Repeat scale scan with varying grid sizes."""
    grid_sizes = [100, 200, 300, 500, 1000]
    results: Dict[str, Any] = {}
    best_scales: List[float] = []

    for ng in grid_sizes:
        r = fit_kernel(kernel_name, x, y,
                       scale_range=scale_range, n_grid=ng,
                       rng_seed=seed, fit_mode=fit_mode)
        best_scales.append(r.log_scale_best)
        results[f"n_grid_{ng}"] = {
            "n_grid": ng,
            "best_log_scale": r.log_scale_best,
            "rms": r.residual_rms,
            "aic": r.aic,
        }

    results["max_delta_log_scale"] = float(np.ptp(best_scales))
    results["within_0p02_dex"] = float(np.ptp(best_scales)) < 0.02
    results["best_log_scale_values"] = best_scales

    return results


# ============================================================
# SUITE E: NEGATIVE CONTROLS / BREAK TESTS
# ============================================================

def run_suite_e_negative_controls(
    x: NDArray, y: NDArray,
    gal_ids: NDArray,
    kernel_name: str,
    baseline_summary: Dict[str, Any],
    n_grid: int = 200,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    fit_mode: str = "direct",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run destructive perturbations and verify signal collapse."""
    results: Dict[str, Any] = {}
    baseline_scale = baseline_summary["best_log_scale"]
    baseline_sharp = baseline_summary.get("scale_scan", {}).get(
        "peak_sharpness", 0)
    baseline_aic_per_n = baseline_summary.get("scale_scan", {}).get(
        "delta_aic_pm_0p1_per_dof", {}).get("mean_0p1", 0)

    log_x = np.log10(x + 1e-300)

    # --- E1: Add noise to log_gbar ---
    for sigma in [0.1, 0.3]:
        label = f"E1_noise_sigma_{sigma}"
        print(f"    {label}...")
        rng = np.random.RandomState(seed + 30000)
        log_x_noisy = log_x + rng.normal(0, sigma, len(log_x))
        x_noisy = 10**log_x_noisy
        # Recompute y = g_obs/g_bar with noisy g_bar
        # y was computed as 10^log_gobs / 10^log_gbar
        # If we add noise to log_gbar, y changes:
        # y_noisy = (g_obs_orig) / (g_bar_noisy)
        # But we don't have g_obs independently. Use the simplest:
        # just fit with x_noisy, y unchanged (noise in predictor)
        r = fit_kernel(kernel_name, x_noisy, y,
                       scale_range=scale_range, n_grid=n_grid,
                       rng_seed=seed, fit_mode=fit_mode)
        scan = scale_injection_scan(
            kernel_name, x_noisy, y,
            log_scale_range=(np.log10(scale_range[0]),
                             np.log10(scale_range[1])),
            n_scan=200, fit_mode=fit_mode,
        )
        results[label] = {
            "sigma_dex": sigma,
            "best_log_scale": r.log_scale_best,
            "delta_from_gdagger": abs(r.log_scale_best - LOG_G_DAGGER),
            "delta_from_baseline": abs(r.log_scale_best - baseline_scale),
            "peak_sharpness": scan.peak_sharpness,
            "sharpness_ratio": (scan.peak_sharpness / baseline_sharp
                                if baseline_sharp > 0 else 0),
            "aic_per_n_0p1": scan.delta_aic_pm_0p1_per_dof.get(
                "mean_0p1", 0),
            "rms": r.residual_rms,
        }

    # --- E2: Warp x via x → x^α ---
    for alpha in [0.7, 1.3]:
        label = f"E2_warp_alpha_{alpha}"
        print(f"    {label}...")
        x_warp = x**alpha
        r = fit_kernel(kernel_name, x_warp, y,
                       scale_range=scale_range, n_grid=n_grid,
                       rng_seed=seed, fit_mode=fit_mode)
        scan = scale_injection_scan(
            kernel_name, x_warp, y,
            log_scale_range=(np.log10(scale_range[0]),
                             np.log10(scale_range[1])),
            n_scan=200, fit_mode=fit_mode,
        )
        results[label] = {
            "alpha": alpha,
            "best_log_scale": r.log_scale_best,
            "delta_from_gdagger": abs(r.log_scale_best - LOG_G_DAGGER),
            "peak_sharpness": scan.peak_sharpness,
            "sharpness_ratio": (scan.peak_sharpness / baseline_sharp
                                if baseline_sharp > 0 else 0),
            "aic_per_n_0p1": scan.delta_aic_pm_0p1_per_dof.get(
                "mean_0p1", 0),
            "rms": r.residual_rms,
        }

    # --- E3: Randomize galaxy association ---
    print("    E3: Galaxy label swap...")
    rng3 = np.random.RandomState(seed + 40000)
    unique_gals = np.unique(gal_ids)
    # Randomly reassign galaxy labels
    shuffled_gals = rng3.permutation(unique_gals)
    gal_map = dict(zip(unique_gals, shuffled_gals))
    # Reconstruct y by mixing points across galaxies
    # Approach: for each galaxy, take another galaxy's y values
    y_swapped = y.copy()
    for g_orig, g_new in gal_map.items():
        if g_orig == g_new:
            continue
        mask_orig = gal_ids == g_orig
        mask_new = gal_ids == g_new
        n_orig = int(np.sum(mask_orig))
        n_new = int(np.sum(mask_new))
        # Take min(n_orig, n_new) points from g_new's y
        n_use = min(n_orig, n_new)
        orig_idx = np.where(mask_orig)[0][:n_use]
        new_idx = np.where(mask_new)[0][:n_use]
        y_swapped[orig_idx] = y[new_idx]

    r = fit_kernel(kernel_name, x, y_swapped,
                   scale_range=scale_range, n_grid=n_grid,
                   rng_seed=seed, fit_mode=fit_mode)
    scan = scale_injection_scan(
        kernel_name, x, y_swapped,
        log_scale_range=(np.log10(scale_range[0]),
                         np.log10(scale_range[1])),
        n_scan=200, fit_mode=fit_mode,
    )
    results["E3_galaxy_swap"] = {
        "best_log_scale": r.log_scale_best,
        "delta_from_gdagger": abs(r.log_scale_best - LOG_G_DAGGER),
        "peak_sharpness": scan.peak_sharpness,
        "sharpness_ratio": (scan.peak_sharpness / baseline_sharp
                            if baseline_sharp > 0 else 0),
        "aic_per_n_0p1": scan.delta_aic_pm_0p1_per_dof.get("mean_0p1", 0),
        "rms": r.residual_rms,
    }

    return results


# ============================================================
# SUITE F: NEARBY-SCALE COMPARISON WITH MATCHED DoF
# ============================================================

def run_suite_f_nearby_scales(
    x: NDArray, y: NDArray,
    kernel_name: str,
    fit_mode: str = "direct",
    n_grid: int = 200,
    scale_range: Tuple[float, float] = (1e-13, 1e-8),
    seed: int = 42,
) -> Dict[str, Any]:
    """Nearby-scale comparison in two modes: η=1 fixed and η-free."""
    kernel_fn = KERNEL_REGISTRY[kernel_name]
    n = len(x)
    results: Dict[str, Any] = {}

    named_scales = {
        "g_dagger": G_DAGGER,
        "a_Lambda": A_LAMBDA,
        "cH0": A_HUBBLE,
        "cH0_over_6": A_HUBBLE / 6.0,
        "cH0_over_2pi": A_HUBBLE / (2.0 * np.pi),
    }

    # --- F1: η fixed (direct substitution, 0 nuisance params) ---
    f1: Dict[str, Any] = {}
    for name, val in named_scales.items():
        _, _, rms = _fit_kernel_at_scale(kernel_fn, x, y, val, fit_mode)
        log_like = -0.5 * n * np.log(2 * np.pi * rms**2 + 1e-300) - n / 2
        n_params = 0 if fit_mode == "direct" else 2  # no scale param
        aic = 2 * n_params - 2 * log_like
        f1[name] = {
            "scale": float(val),
            "log_scale": float(np.log10(val)),
            "rms": float(rms),
            "aic": float(aic),
            "n_params": n_params,
        }
    f1["interpretation"] = (
        "F1 tests each scale with eta=1 (direct substitution). "
        "A large DELTA_AIC(a_Lambda vs g_dagger) proves eta!=1, "
        "NOT that Lambda cannot set the scale."
    )
    results["F1_eta_fixed"] = f1

    # --- F2: η free (fit amplitude A, i.e., y = A * kernel(x, named_scale)) ---
    f2: Dict[str, Any] = {}
    for name, val in named_scales.items():
        # Fit y = A * kernel(x, val) with A free (1 param)
        k_vals = kernel_fn(x, val)
        if not np.all(np.isfinite(k_vals)):
            f2[name] = {"error": "non-finite kernel values"}
            continue
        # OLS for A: y = A*k → A = sum(y*k)/sum(k^2)
        A_hat = float(np.sum(y * k_vals) / (np.sum(k_vals**2) + 1e-300))
        resid = y - A_hat * k_vals
        rms = float(np.sqrt(np.mean(resid**2)))
        n_params = 1  # eta (amplitude) free
        log_like = -0.5 * n * np.log(2 * np.pi * rms**2 + 1e-300) - n / 2
        aic = 2 * n_params - 2 * log_like
        f2[name] = {
            "scale": float(val),
            "log_scale": float(np.log10(val)),
            "eta_hat": A_hat,
            "rms": float(rms),
            "aic": float(aic),
            "n_params": n_params,
        }
    f2["interpretation"] = (
        "F2 tests each scale with eta free (y = eta * kernel(x, scale)). "
        "All models have matched DoF (1 param each). "
        "This is a fairer comparison than F1."
    )
    results["F2_eta_free"] = f2

    # --- Also run free-scale fit for reference ---
    r_free = fit_kernel(kernel_name, x, y,
                        scale_range=scale_range, n_grid=n_grid,
                        rng_seed=seed, fit_mode=fit_mode)
    results["free_scale_reference"] = {
        "best_log_scale": r_free.log_scale_best,
        "best_scale": r_free.scale_best,
        "rms": r_free.residual_rms,
        "aic": r_free.aic,
        "n_params": r_free.n_params,
    }

    return results


# ============================================================
# SUITE G: NON-RAR PILOT
# ============================================================

def run_suite_g_nonrar_pilot(
    scale_range: Tuple[float, float] = (1e-12, 1e-8),
    n_grid: int = 200,
    n_shuffles: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run cluster RAR as explicit non-RAR pilot (expected null)."""
    try:
        lg, lt, eg, et = load_tian2020_cluster_rar(
            project_root=PROJECT_ROOT)
    except FileNotFoundError as e:
        return {"skipped": True, "reason": str(e)}

    x = 10**lg
    y = 10**lt / 10**lg
    n = len(x)

    config = ExperimentConfig(
        tag="cluster_nonrar_pilot",
        seed=seed,
        n_shuffles=n_shuffles,
        n_grid=n_grid,
        n_scan=200,
        scale_range=scale_range,
        fit_mode="log",
    )
    summary = run_experiment(
        x, y, config,
        dataset_name="Tian2020_Clusters_NonRAR",
        dataset_meta={
            "n_points": n,
            "source": "Tian+2020 CLASH",
            "note": "Expected NULL — cluster a0 ≈ 14× g†",
        },
    )

    return {
        "dataset": "Tian2020_Clusters",
        "n_data": n,
        "best_kernel": summary.get("best_kernel_name", ""),
        "best_log_scale": summary.get("best_log_scale", -99),
        "delta_from_gdagger": abs(
            summary.get("best_log_scale", -99) - LOG_G_DAGGER),
        "verdicts": summary.get("verdicts", {}),
        "output_dir": summary.get("output_dir", ""),
        "note": "Cluster-scale RAR expected to NOT find g†",
    }


# ============================================================
# FIGURE GENERATORS
# ============================================================

def make_figure_1_three_panel(run_dir: str, summary: Dict[str, Any],
                               baseline: Dict[str, Any]) -> str:
    """Fig 1: AIC curve + null histogram + ΔAIC/N bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): AIC vs log-scale
    ax1 = axes[0]
    scan = baseline.get("scale_scan", {})
    if scan:
        log_s = np.array(scan["log_scales"])
        aic = np.array(scan["aic_values"])
        aic_best = np.min(aic)
        ax1.plot(log_s, aic - aic_best, "k-", lw=1.2)
        best_ls = scan["best_log_scale"]
        ax1.axvline(best_ls, color="green", ls="-.", lw=1.5,
                     label=f"best = {best_ls:.3f}")
        ax1.axvline(LOG_G_DAGGER, color="red", ls="-", lw=1.8,
                     label=f"g† = {LOG_G_DAGGER:.3f}")
        ax1.axvline(np.log10(A_HUBBLE / 6.0), color="darkorange", ls="--",
                     lw=1.5, label=f"cH₀/6 = {np.log10(A_HUBBLE/6):.3f}")
        ax1.axvline(np.log10(A_LAMBDA), color="purple", ls=":", lw=1.5,
                     label=f"a_Λ = {np.log10(A_LAMBDA):.3f}")
        ax1.set_xlim(best_ls - 1.5, best_ls + 1.5)
        mask = (log_s > best_ls - 1.5) & (log_s < best_ls + 1.5)
        y_max = float(np.max((aic - aic_best)[mask]))
        ax1.set_ylim(-5, min(y_max * 1.1, 2000))
        ax1.set_xlabel("log₁₀(scale / m s⁻²)", fontsize=10)
        ax1.set_ylabel("ΔAIC (from best)", fontsize=10)
        ax1.set_title("(a) Scale preference — SPARC RAR", fontsize=11,
                        fontweight="bold")
        ax1.legend(fontsize=7, loc="upper left")

    # Panel (b): Null distributions (destructive nulls only + control)
    ax2 = axes[1]
    null_data = summary.get("suite_a", {})
    for label, color, key in [
        ("Global null", "gray", "A1_global"),
        ("Block-perm null", "steelblue", "A2b_block_permute_bins"),
        ("Galaxy-shift null", "darkorange", "A3_within_galaxy"),
        ("Bin control", "lightgreen", "A2_within_bin"),
    ]:
        null_scales = null_data.get(key, {}).get("null_best_log_scales", [])
        if null_scales:
            ax2.hist(null_scales, bins=30, alpha=0.35, color=color,
                     label=label, edgecolor=color)

    obs_scale = null_data.get("A1_global", {}).get(
        "observed_best_log_scale",
        baseline.get("best_log_scale", LOG_G_DAGGER))
    ax2.axvline(LOG_G_DAGGER, color="red", ls="-", lw=1.8,
                 label=f"g† = {LOG_G_DAGGER:.3f}")
    ax2.axvspan(LOG_G_DAGGER - 0.1, LOG_G_DAGGER + 0.1,
                 alpha=0.08, color="red")

    # Get p-values for destructive nulls only
    p_vals = []
    for key, lbl in [("A1_global", "G"), ("A2b_block_permute_bins", "B"),
                      ("A3_within_galaxy", "W")]:
        p = null_data.get(key, {}).get("p_within_0p1_dex", -1)
        if p >= 0:
            p_vals.append(f"{lbl}={p:.3f}")
    p_str = ", ".join(p_vals) if p_vals else "N/A"

    ax2.set_xlabel("log₁₀(best scale from null)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title(f"(b) Null distributions — p: {p_str}", fontsize=11,
                    fontweight="bold")
    ax2.legend(fontsize=7, loc="upper left")

    # Panel (c): ΔAIC/N bar chart
    ax3 = axes[2]
    # Collect from baseline + cluster
    datasets = {"SPARC\nRAR": baseline}
    suite_g = summary.get("suite_g", {})
    if not suite_g.get("skipped"):
        datasets["Cluster\n(null)"] = suite_g

    labels = []
    means = []
    for dname, d in datasets.items():
        per_dof = d.get("scale_scan", {}).get("delta_aic_pm_0p1_per_dof", {})
        m = per_dof.get("mean_0p1", 0)
        labels.append(dname)
        means.append(m)

    if labels:
        colors = ["steelblue" if m > 0.01 else "lightcoral" for m in means]
        ax3.bar(range(len(labels)), means, color=colors, alpha=0.85)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylabel("ΔAIC/N at ±0.1 dex", fontsize=10)
        ax3.set_title("(c) Scale identifiability", fontsize=11,
                        fontweight="bold")
        ax3.axhline(0, color="gray", lw=0.5)

    fig.tight_layout()
    path = os.path.join(run_dir, "figures", "fig1_three_panel.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figure_2_validation_stability(
    run_dir: str, summary: Dict[str, Any]
) -> str:
    """Fig 2: GroupKFold + cut sensitivity + grid invariance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): GroupKFold fold-by-fold scale
    ax1 = axes[0]
    cv_data = summary.get("suite_b", {})
    fold_details = cv_data.get("fold_details", [])
    if fold_details:
        folds = [d["fold"] for d in fold_details]
        scales = [d["best_log_scale"] for d in fold_details]
        ax1.bar(folds, scales, color="steelblue", alpha=0.85,
                bottom=0, zorder=3)
        # Zoom y-axis to show structure near g†
        scale_arr = np.array(scales)
        y_center = np.mean(scale_arr)
        y_range = max(np.ptp(scale_arr) * 2, 0.3)
        ax1.set_ylim(y_center - y_range, y_center + y_range)
        ax1.axhline(LOG_G_DAGGER, color="red", ls="--", lw=1.5,
                     label=f"g† = {LOG_G_DAGGER:.3f}")
        ax1.axhspan(LOG_G_DAGGER - 0.1, LOG_G_DAGGER + 0.1,
                     alpha=0.1, color="red")
        ax1.set_xlabel("Fold", fontsize=10)
        ax1.set_ylabel("Best log₁₀(scale)", fontsize=10)
        ax1.set_title(
            f"(a) GroupKFold — "
            f"μ={cv_data.get('best_log_scale_mean', 0):.3f} "
            f"± {cv_data.get('best_log_scale_std', 0):.3f}",
            fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)

    # Panel (b): Cut sensitivity
    ax2 = axes[1]
    cut_data = summary.get("suite_c", {})
    cut_labels = []
    cut_scales = []
    for name in ["baseline", "strict_inc", "strict_quality", "strict_npts"]:
        d = cut_data.get(name, {})
        if d:
            cut_labels.append(name.replace("strict_", "").replace("_", "\n"))
            cut_scales.append(d.get("best_log_scale", 0))
    if cut_labels:
        colors = ["steelblue" if abs(s - LOG_G_DAGGER) < 0.1
                   else "lightcoral" for s in cut_scales]
        ax2.bar(range(len(cut_labels)), cut_scales, color=colors, alpha=0.85,
                zorder=3)
        # Zoom y-axis
        cs_arr = np.array(cut_scales)
        y_center = np.mean(cs_arr)
        y_range = max(np.ptp(cs_arr) * 2, 0.3)
        ax2.set_ylim(y_center - y_range, y_center + y_range)
        ax2.axhline(LOG_G_DAGGER, color="red", ls="--", lw=1.5,
                     label=f"g† = {LOG_G_DAGGER:.3f}")
        ax2.axhspan(LOG_G_DAGGER - 0.1, LOG_G_DAGGER + 0.1,
                     alpha=0.1, color="red")
        ax2.set_xticks(range(len(cut_labels)))
        ax2.set_xticklabels(cut_labels, fontsize=9)
        ax2.set_ylabel("Best log₁₀(scale)", fontsize=10)
        ax2.set_title(
            f"(b) Cut sensitivity — "
            f"Δmax = {cut_data.get('max_delta_log_scale', 0):.4f} dex",
            fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)

    # Panel (c): Grid invariance
    ax3 = axes[2]
    grid_data = summary.get("suite_d", {})
    grid_labels = []
    grid_scales = []
    for ng in [100, 200, 300, 500, 1000]:
        d = grid_data.get(f"n_grid_{ng}", {})
        if d:
            grid_labels.append(str(ng))
            grid_scales.append(d.get("best_log_scale", 0))
    if grid_labels:
        ax3.plot(range(len(grid_labels)), grid_scales, "o-",
                 color="steelblue", lw=1.5, ms=8)
        ax3.axhline(LOG_G_DAGGER, color="red", ls="--", lw=1.5,
                     label=f"g† = {LOG_G_DAGGER:.3f}")
        ax3.axhspan(LOG_G_DAGGER - 0.1, LOG_G_DAGGER + 0.1,
                     alpha=0.1, color="red")
        ax3.set_xticks(range(len(grid_labels)))
        ax3.set_xticklabels(grid_labels, fontsize=9)
        ax3.set_xlabel("Grid size (n_grid)", fontsize=10)
        ax3.set_ylabel("Best log₁₀(scale)", fontsize=10)
        ax3.set_title(
            f"(c) Grid invariance — "
            f"Δmax = {grid_data.get('max_delta_log_scale', 0):.4f} dex",
            fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(run_dir, "figures", "fig2_validation_stability.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figure_3_negative_controls(
    run_dir: str, summary: Dict[str, Any]
) -> str:
    """Fig 3: Negative controls — sharpness collapse under perturbation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    neg = summary.get("suite_e", {})

    # Panel (a): Sharpness ratio vs perturbation
    ax1 = axes[0]
    labels = []
    sharp_ratios = []
    delta_from_gd = []

    controls = [
        ("E1_noise_sigma_0.1", "Noise σ=0.1"),
        ("E1_noise_sigma_0.3", "Noise σ=0.3"),
        ("E2_warp_alpha_0.7", "Warp α=0.7"),
        ("E2_warp_alpha_1.3", "Warp α=1.3"),
        ("E3_galaxy_swap", "Galaxy swap"),
    ]
    for key, label in controls:
        d = neg.get(key, {})
        if d:
            labels.append(label)
            sharp_ratios.append(d.get("sharpness_ratio", 0))
            delta_from_gd.append(d.get("delta_from_gdagger", 0))

    if labels:
        colors = ["lightcoral" if r < 0.5 else "steelblue"
                   for r in sharp_ratios]
        ax1.barh(range(len(labels)), sharp_ratios, color=colors, alpha=0.85)
        ax1.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.5,
                     label="Baseline")
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel("Peak sharpness / baseline", fontsize=10)
        ax1.set_title("(a) Sharpness collapse under perturbation",
                        fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)

    # Panel (b): Scale drift (Δ from g†) vs perturbation
    ax2 = axes[1]
    if labels:
        colors2 = ["lightcoral" if d > 0.3 else "steelblue"
                    for d in delta_from_gd]
        ax2.barh(range(len(labels)), delta_from_gd, color=colors2, alpha=0.85)
        ax2.axvline(0.1, color="red", ls="--", lw=1.5,
                     label="±0.1 dex threshold")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel("|Δ log₁₀ scale − log₁₀ g†| (dex)", fontsize=10)
        ax2.set_title("(b) Scale drift under perturbation",
                        fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(run_dir, "figures", "fig3_negative_controls.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="g† Hunt Adversarial Validation Suite")
    parser.add_argument("--n-shuffles", type=int, default=500,
                        help="Shuffles per null test (default 500)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (100 shuffles, reduced grids)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "outputs", "gdagger_hunt"))
    args = parser.parse_args()

    if args.quick:
        args.n_shuffles = 100
        n_grid = 100
        print("*** QUICK MODE (100 shuffles, n_grid=100) ***\n")
    else:
        n_grid = 200

    seed = args.seed
    n_shuffles = args.n_shuffles
    scale_range = (1e-13, 1e-8)
    fit_mode = "direct"

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, f"{ts}_refereeproof")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  g† HUNT — ADVERSARIAL VALIDATION SUITE")
    print(f"  Output: {run_dir}")
    print(f"  Shuffles: {n_shuffles}, Grid: {n_grid}, Seed: {seed}")
    print(f"{'='*70}\n")

    t_total = time.time()
    full_summary: Dict[str, Any] = {
        "timestamp": ts,
        "seed": seed,
        "n_shuffles": n_shuffles,
        "n_grid": n_grid,
        "fit_mode": fit_mode,
    }

    # -------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------
    print("Loading SPARC RAR data with galaxy IDs...")
    log_gbar, log_gobs, gal_ids, gal_names = load_sparc_rar_with_galaxies(
        project_root=PROJECT_ROOT)
    x = 10**log_gbar
    y = 10**log_gobs / 10**log_gbar

    print(f"  N = {len(x)} points, {len(gal_names)} galaxies")
    print(f"  x (g_bar): [{x.min():.2e}, {x.max():.2e}] m/s²")
    print(f"  y (g_obs/g_bar): [{y.min():.3f}, {y.max():.3f}]")

    full_summary["data"] = {
        "n_points": len(x),
        "n_galaxies": len(gal_names),
        "galaxy_names": gal_names,
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
    }

    # -------------------------------------------------------
    # BASELINE: Full SPARC RAR control
    # -------------------------------------------------------
    print("\n--- BASELINE: SPARC RAR control ---")
    baseline_config = ExperimentConfig(
        tag="baseline_sparc_rar",
        seed=seed,
        n_shuffles=min(50, n_shuffles),  # smaller for baseline
        n_grid=n_grid,
        n_scan=300,
        scale_range=scale_range,
        fit_mode=fit_mode,
    )
    baseline = run_experiment(
        x, y, baseline_config,
        dataset_name="SPARC_RAR_baseline",
        dataset_meta={"n_points": len(x), "n_galaxies": len(gal_names)},
    )
    full_summary["baseline"] = baseline
    best_kernel = baseline.get("best_kernel_name", "BE_RAR")
    print(f"  Best kernel: {best_kernel}")
    print(f"  Best log scale: {baseline.get('best_log_scale', 'N/A')}")
    print(f"  Δ from g†: {abs(baseline.get('best_log_scale', 0) - LOG_G_DAGGER):.4f} dex")

    # -------------------------------------------------------
    # SUITE A: SHUFFLE NULLS
    # -------------------------------------------------------
    print(f"\n--- SUITE A: SHUFFLE NULLS ({n_shuffles} each) ---")
    t0 = time.time()
    suite_a = run_suite_a_shuffle_nulls(
        x, y, gal_ids, best_kernel,
        n_shuffles=n_shuffles, seed=seed,
        scale_range=scale_range, n_grid=n_grid,
        fit_mode=fit_mode,
    )
    suite_a["total_runtime_s"] = time.time() - t0
    full_summary["suite_a"] = suite_a
    print(f"  Suite A complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE B: GROUPED CROSS-VALIDATION
    # -------------------------------------------------------
    print("\n--- SUITE B: GROUPED CROSS-VALIDATION ---")
    t0 = time.time()
    suite_b = run_suite_b_grouped_cv(
        x, y, gal_ids, best_kernel,
        n_folds=5, seed=seed,
        scale_range=scale_range, n_grid=n_grid,
        fit_mode=fit_mode,
    )
    suite_b["runtime_s"] = time.time() - t0
    full_summary["suite_b"] = suite_b
    print(f"  Kernel unanimity: {suite_b.get('best_kernel_unanimous', False)}")
    print(f"  Scale: {suite_b.get('best_log_scale_mean', 0):.3f} "
          f"± {suite_b.get('best_log_scale_std', 0):.3f}")
    print(f"  Suite B complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE C: CUT SENSITIVITY
    # -------------------------------------------------------
    print("\n--- SUITE C: CUT / SAMPLE SENSITIVITY ---")
    t0 = time.time()
    suite_c = run_suite_c_cut_sensitivity(
        best_kernel, scale_range=scale_range, n_grid=n_grid,
        fit_mode=fit_mode, seed=seed,
    )
    suite_c["runtime_s"] = time.time() - t0
    full_summary["suite_c"] = suite_c
    print(f"  Max Δ(log scale): {suite_c.get('max_delta_log_scale', 0):.4f} dex")
    print(f"  Suite C complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE D: GRID INVARIANCE
    # -------------------------------------------------------
    print("\n--- SUITE D: GRID / OPTIMIZATION INVARIANCE ---")
    t0 = time.time()
    suite_d = run_suite_d_grid_invariance(
        x, y, best_kernel,
        scale_range=scale_range, fit_mode=fit_mode, seed=seed,
    )
    suite_d["runtime_s"] = time.time() - t0
    full_summary["suite_d"] = suite_d
    print(f"  Max Δ(log scale): {suite_d.get('max_delta_log_scale', 0):.4f} dex")
    print(f"  Within 0.02 dex: {suite_d.get('within_0p02_dex', False)}")
    print(f"  Suite D complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE E: NEGATIVE CONTROLS
    # -------------------------------------------------------
    print("\n--- SUITE E: NEGATIVE CONTROLS ---")
    t0 = time.time()
    suite_e = run_suite_e_negative_controls(
        x, y, gal_ids, best_kernel, baseline,
        n_grid=n_grid, scale_range=scale_range,
        fit_mode=fit_mode, seed=seed,
    )
    suite_e["runtime_s"] = time.time() - t0
    full_summary["suite_e"] = suite_e
    print(f"  Suite E complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE F: NEARBY-SCALE COMPARISON
    # -------------------------------------------------------
    print("\n--- SUITE F: NEARBY-SCALE COMPARISON ---")
    t0 = time.time()
    suite_f = run_suite_f_nearby_scales(
        x, y, best_kernel,
        fit_mode=fit_mode, n_grid=n_grid,
        scale_range=scale_range, seed=seed,
    )
    suite_f["runtime_s"] = time.time() - t0
    full_summary["suite_f"] = suite_f
    print(f"  Suite F complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # SUITE G: NON-RAR PILOT
    # -------------------------------------------------------
    print("\n--- SUITE G: NON-RAR PILOT ---")
    t0 = time.time()
    suite_g = run_suite_g_nonrar_pilot(
        scale_range=(1e-12, 1e-8), n_grid=n_grid,
        n_shuffles=min(100, n_shuffles), seed=seed,
    )
    suite_g["runtime_s"] = time.time() - t0
    full_summary["suite_g"] = suite_g
    print(f"  Suite G complete ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------
    # GENERATE FIGURES
    # -------------------------------------------------------
    print("\n--- GENERATING FIGURES ---")
    fig1_path = make_figure_1_three_panel(run_dir, full_summary, baseline)
    print(f"  Fig 1: {fig1_path}")

    fig2_path = make_figure_2_validation_stability(run_dir, full_summary)
    print(f"  Fig 2: {fig2_path}")

    fig3_path = make_figure_3_negative_controls(run_dir, full_summary)
    print(f"  Fig 3: {fig3_path}")

    full_summary["figures"] = {
        "fig1_three_panel": fig1_path,
        "fig2_validation_stability": fig2_path,
        "fig3_negative_controls": fig3_path,
    }

    # -------------------------------------------------------
    # REPRODUCIBILITY METADATA
    # -------------------------------------------------------
    print("\n--- REPRODUCIBILITY METADATA ---")
    repro: Dict[str, Any] = {"seed": seed, "n_shuffles": n_shuffles}

    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        repro["git_commit"] = git_hash
    except Exception:
        repro["git_commit"] = "unavailable"

    # Pip freeze
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL
        ).decode()
        repro_path = os.path.join(run_dir, "pip_freeze.txt")
        with open(repro_path, "w") as f:
            f.write(freeze)
        repro["pip_freeze_path"] = repro_path
    except Exception:
        repro["pip_freeze_path"] = "unavailable"

    full_summary["reproducibility"] = repro
    full_summary["total_runtime_s"] = time.time() - t_total

    # -------------------------------------------------------
    # WRITE SUMMARY
    # -------------------------------------------------------
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    # SHA256 hashes
    hashes: Dict[str, str] = {}
    for fpath in [summary_path, fig1_path, fig2_path, fig3_path]:
        if os.path.exists(fpath):
            h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            hashes[os.path.basename(fpath)] = h
    with open(os.path.join(run_dir, "sha256_hashes.json"), "w") as f:
        json.dump(hashes, f, indent=2)
    full_summary["sha256_hashes"] = hashes

    # Re-write summary with hashes
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    # -------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  ADVERSARIAL VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Run folder: {run_dir}")
    print(f"  Total time: {full_summary['total_runtime_s']:.1f}s")
    print(f"\n  HEADLINE NUMBERS:")
    print(f"    Best kernel: {best_kernel}")
    print(f"    Best log scale: {baseline.get('best_log_scale', 'N/A'):.4f}")
    print(f"    Δ from g†: {abs(baseline.get('best_log_scale', 0) - LOG_G_DAGGER):.4f} dex")

    scan = baseline.get("scale_scan", {})
    print(f"    ΔAIC/N at ±0.1 dex: "
          f"{scan.get('delta_aic_pm_0p1_per_dof', {}).get('mean_0p1', 0):.4f}")
    print(f"    Peak sharpness: {scan.get('peak_sharpness', 0):.0f}")

    print(f"\n  Permutation tests (dual-window ±0.05 / ±0.10 dex):")
    for key, label in [("A1_global", "Global null"),
                        ("A2b_block_permute_bins", "Block-perm null"),
                        ("A2_within_bin", "Bin control"),
                        ("A3_within_galaxy", "Galaxy null")]:
        d = suite_a.get(key, {})
        p10 = d.get("p_within_0p10_dex", d.get("p_within_0p1_dex", -1))
        p05 = d.get("p_within_0p05_dex", -1)
        n10 = d.get("n_hits_0p10", d.get("n_hits", "?"))
        n05 = d.get("n_hits_0p05", "?")
        up = d.get("p_upper_95", None)
        line = f"    {label:18s}: ±0.10={p10:.4f} ({n10}/{n_shuffles})"
        line += f"  ±0.05={p05:.4f} ({n05}/{n_shuffles})"
        if up is not None:
            line += f"  [CP UB: {up:.4f}]"
        print(line)

    print(f"    GroupKFold scale: "
          f"{suite_b.get('best_log_scale_mean', 0):.3f} "
          f"± {suite_b.get('best_log_scale_std', 0):.3f}")
    print(f"    Cut max Δ: {suite_c.get('max_delta_log_scale', 0):.4f} dex")
    print(f"    Grid max Δ: {suite_d.get('max_delta_log_scale', 0):.4f} dex")

    print(f"\n  Negative control degradation:")
    for key, label in [("E1_noise_sigma_0.1", "Noise 0.1"),
                        ("E1_noise_sigma_0.3", "Noise 0.3"),
                        ("E2_warp_alpha_0.7", "Warp 0.7"),
                        ("E2_warp_alpha_1.3", "Warp 1.3"),
                        ("E3_galaxy_swap", "Gal swap")]:
        d = suite_e.get(key, {})
        sr = d.get("sharpness_ratio", 0)
        dg = d.get("delta_from_gdagger", 0)
        print(f"    {label:12s}: sharpness {sr:.3f}× baseline, "
              f"Δg† = {dg:.3f} dex")

    print(f"\n  Files in run folder:")
    for root, dirs, files in os.walk(run_dir):
        for fname in sorted(files):
            rel = os.path.relpath(os.path.join(root, fname), run_dir)
            print(f"    {rel}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
