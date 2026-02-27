#!/usr/bin/env python3
"""
test_gdagger_hunt_pilot.py — Pilot end-to-end test for the g† Hunt module.

Runs three experiments:
  1. SYNTHETIC SANITY: Generate data from known BE kernel + known g†,
     verify recovery of kernel identity and scale within ±0.05 dex,
     verify shuffles destroy the signal.
  2. RAR CONTROL: Use SPARC rotation curve data to confirm the tool
     rediscovers g† and the BE kernel in the known context.
  3. CLUSTER PILOT: Use Tian+2020 CLASH cluster RAR data as a
     non-rotation-curve context. This is an independent physical
     regime (gravitational lensing + X-ray hydrostatic masses).

Each experiment produces:
  - outputs/gdagger_hunt/YYYYMMDD_HHMMSS_<tag>/
      params.json, summary.json, metrics.parquet, figures/, logs.txt

Acceptance criteria:
  - Synthetic: recovers correct kernel and g† within ±0.05 dex
  - Synthetic: shuffle null p(near g†) > 0.1
  - RAR control: finds optimum within ±0.1 dex of g†
  - RAR control: prefers BE-family kernel over control kernels
  - Code produces all outputs without errors

Usage:
  python3 analysis/tests/test_gdagger_hunt_pilot.py
  python3 analysis/tests/test_gdagger_hunt_pilot.py --n-shuffles 1000 --parallel
  python3 analysis/tests/test_gdagger_hunt_pilot.py --smoke  # fast mode

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from gdagger_hunt import (
    ExperimentConfig,
    G_DAGGER,
    LOG_G_DAGGER,
    KERNEL_REGISTRY,
    generate_synthetic,
    load_sparc_rar,
    load_tian2020_cluster_rar,
    run_experiment,
    generate_pi_groups,
    STANDARD_CONSTANTS,
    PhysicalQuantity,
    A_LAMBDA,
    A_HUBBLE,
    plot_three_panel_summary,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_verdict(name: str, passed: bool, detail: str = "") -> None:
    """Print a verdict line."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}: {detail}")


def run_pi_group_demo() -> None:
    """Demonstrate the dimensionless-group generator."""
    print_header("DIMENSIONLESS GROUP GENERATOR DEMO")

    # Target: acceleration dimensions [M=0, L=1, T=-2, K=0]
    target_dims = (0, 1, -2, 0)
    print(f"Target dimensions: [M, L, T, K] = {target_dims}  (acceleration)")
    print(f"Known scales: g† = {G_DAGGER:.2e}, a_Λ = {A_LAMBDA:.2e}, "
          f"cH₀ = {A_HUBBLE:.2e}\n")

    groups = generate_pi_groups(
        target_dims=target_dims,
        constants=STANDARD_CONSTANTS,
        exponent_set=(-2, -1, -0.5, 0, 0.5, 1, 2),
        max_constants=3,
        require_lambda=False,
    )

    print(f"Found {len(groups)} candidate acceleration scales:\n")
    for i, g in enumerate(groups[:15]):  # Show top 15
        ratio = g["numeric_value"] / G_DAGGER
        print(f"  {i+1:2d}. {g['formula']:40s}  "
              f"= {g['numeric_value']:.3e}  "
              f"({ratio:.2f}× g†)  "
              f"{'[uses Λ]' if g['uses_lambda'] else ''}")

    # Find the a_Lambda group
    lambda_groups = [g for g in groups if g["uses_lambda"]]
    if lambda_groups:
        best_lambda = min(lambda_groups,
                          key=lambda g: abs(np.log10(
                              g["numeric_value"] / G_DAGGER)))
        ratio = best_lambda["numeric_value"] / G_DAGGER
        print(f"\n  Closest Λ-based scale to g†: {best_lambda['formula']}")
        print(f"  Value: {best_lambda['numeric_value']:.3e} "
              f"= {ratio:.2f}× g†")
        print(f"  → η = g† / a_Λ = {G_DAGGER / A_LAMBDA:.4f}")

    print()


def run_synthetic_test(config: ExperimentConfig) -> dict:
    """Experiment 1: Synthetic sanity check."""
    print_header("EXPERIMENT 1: SYNTHETIC SANITY CHECK")

    # Generate synthetic data from BE_RAR kernel with known g†
    print("Generating synthetic BE_RAR data with true g† = "
          f"{G_DAGGER:.2e}...")
    x, y, truth = generate_synthetic(
        kernel_name="BE_RAR",
        true_scale=G_DAGGER,
        n_points=500,
        x_range=(1e-13, 1e-9),
        amplitude=1.0,
        offset=0.0,
        noise_sigma=0.05,
        seed=config.seed,
    )
    print(f"  N = {len(x)}, x range: [{x.min():.2e}, {x.max():.2e}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    synth_config = ExperimentConfig(
        tag="synthetic_sanity",
        seed=config.seed,
        n_shuffles=config.n_shuffles,
        n_grid=config.n_grid,
        n_scan=config.n_scan,
        parallel=config.parallel,
        n_workers=config.n_workers,
        output_base=config.output_base,
    )

    summary = run_experiment(
        x, y, synth_config,
        dataset_name="Synthetic_BE_RAR",
        dataset_meta=truth,
    )

    # Check results
    print("\n  Results:")
    best = summary.get("best_kernel_name", "NONE")
    best_log_s = summary.get("best_log_scale", -99)
    delta = abs(best_log_s - LOG_G_DAGGER)

    print_verdict(
        "Kernel identity",
        best in ("BE_RAR", "BE_cousin", "BE_occupation"),
        f"Best = {best} (expected BE family)"
    )
    print_verdict(
        "Scale recovery",
        delta < 0.05,
        f"Δ(log scale) = {delta:.4f} dex (threshold: 0.05)"
    )

    if "shuffle_null" in summary:
        p_rms_synth = summary["shuffle_null"]["p_value_rms"]
        p_near = summary["shuffle_null"]["p_value_near_gdagger"]
        # For synthetic with real signal: shuffles should have WORSE fit
        # and should NOT converge on g†. So p_rms should be low and
        # p_near should be low (null doesn't reproduce the scale preference).
        print_verdict(
            "Shuffle fit worse than real",
            p_rms_synth < 0.05,
            f"p(rms) = {p_rms_synth:.3f} (should be < 0.05)"
        )
        print_verdict(
            "Shuffle doesn't find g†",
            p_near < 0.10,
            f"p(null near g†) = {p_near:.3f} (should be < 0.10)"
        )

    if "scale_scan" in summary:
        within = summary["scale_scan"]["within_0p1_dex"]
        print_verdict(
            "Scale scan peaks near g†",
            within,
            f"Best scan scale = {summary['scale_scan']['best_log_scale']:.3f}"
        )

    print(f"\n  Output: {summary['output_dir']}")
    return summary


def run_rar_control(config: ExperimentConfig) -> dict:
    """Experiment 2: SPARC RAR as control (should rediscover g†)."""
    print_header("EXPERIMENT 2: SPARC RAR CONTROL")

    try:
        log_gbar, log_gobs = load_sparc_rar(project_root=PROJECT_ROOT)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return {"skipped": True, "reason": str(e)}

    print(f"  Loaded {len(log_gbar)} RAR points from SPARC")

    # Transform to (x, y) = (g_bar, g_obs/g_bar) — the mapping ratio
    x = 10**log_gbar
    y = 10**log_gobs / 10**log_gbar

    print(f"  x (g_bar): [{x.min():.2e}, {x.max():.2e}] m/s²")
    print(f"  y (g_obs/g_bar): [{y.min():.3f}, {y.max():.3f}]")

    rar_config = ExperimentConfig(
        tag="sparc_rar_control",
        seed=config.seed,
        n_shuffles=config.n_shuffles,
        n_grid=config.n_grid,
        n_scan=config.n_scan,
        parallel=config.parallel,
        n_workers=config.n_workers,
        output_base=config.output_base,
        fit_mode="direct",  # RAR mapping IS the kernel (y = K(x,s))
    )

    summary = run_experiment(
        x, y, rar_config,
        dataset_name="SPARC_RAR",
        dataset_meta={"n_points": len(x), "source": "SPARC table2",
                      "transform": "y = g_obs / g_bar"},
    )

    # Check results
    print("\n  Results:")
    best = summary.get("best_kernel_name", "NONE")
    best_log_s = summary.get("best_log_scale", -99)
    delta = abs(best_log_s - LOG_G_DAGGER)

    print_verdict(
        "Prefers BE-family kernel",
        best in ("BE_RAR", "BE_cousin", "BE_occupation", "coth"),
        f"Best = {best}"
    )
    print_verdict(
        "Scale within ±0.1 dex of g†",
        delta < 0.1,
        f"Δ = {delta:.4f} dex (log scale = {best_log_s:.3f})"
    )

    if "shuffle_null" in summary:
        p_rms = summary["shuffle_null"]["p_value_rms"]
        print_verdict(
            "Fit better than shuffled",
            p_rms < 0.05,
            f"p(rms) = {p_rms:.4f}"
        )

    # AIC comparison
    if "kernel_ranking" in summary:
        ranking = summary["kernel_ranking"]
        be_names = {"BE_RAR", "BE_cousin", "BE_occupation"}
        be_results = [r for r in ranking if r["kernel_name"] in be_names]
        nonbe_results = [r for r in ranking if r["kernel_name"] not in be_names]

        if be_results and nonbe_results:
            best_be_aic = min(r["aic"] for r in be_results)
            best_nonbe_aic = min(r["aic"] for r in nonbe_results)
            delta_aic = best_nonbe_aic - best_be_aic
            print_verdict(
                "ΔAIC(non-BE − BE) > 0",
                delta_aic > 0,
                f"ΔAIC = {delta_aic:.1f} (positive = BE preferred)"
            )

    print(f"\n  Output: {summary['output_dir']}")
    return summary


def run_cluster_pilot(config: ExperimentConfig) -> dict:
    """Experiment 3: Tian+2020 cluster RAR (non-rotation-curve context)."""
    print_header("EXPERIMENT 3: CLUSTER RAR PILOT (Tian+2020)")

    try:
        lg, lt, eg, et = load_tian2020_cluster_rar(project_root=PROJECT_ROOT)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return {"skipped": True, "reason": str(e)}

    print(f"  Loaded {len(lg)} cluster RAR points (20 CLASH clusters)")

    # Transform: x = g_bar (baryonic accel), y = g_tot / g_bar (mapping ratio)
    x = 10**lg
    y = 10**lt / 10**lg

    print(f"  x (g_bar): [{x.min():.2e}, {x.max():.2e}] m/s²")
    print(f"  y (g_tot/g_bar): [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Note: Cluster a₀ ≈ 14× g† — a DIFFERENT scale is expected")

    cluster_config = ExperimentConfig(
        tag="cluster_rar_pilot",
        seed=config.seed,
        n_shuffles=config.n_shuffles,
        n_grid=config.n_grid,
        n_scan=config.n_scan,
        scale_range=(1e-12, 1e-8),  # Wider range for clusters
        parallel=config.parallel,
        n_workers=config.n_workers,
        output_base=config.output_base,
        fit_mode="log",  # Cluster RAR mapping is also multiplicative
    )

    summary = run_experiment(
        x, y, cluster_config,
        dataset_name="Tian2020_CLASH_Clusters",
        dataset_meta={
            "n_points": len(x),
            "source": "Tian+2020 VizieR J/ApJ/896/70",
            "transform": "y = g_tot / g_bar",
            "note": "Cluster-scale RAR, expect a0 ~ 14x g†",
        },
    )

    # Check results
    print("\n  Results:")
    best = summary.get("best_kernel_name", "NONE")
    best_log_s = summary.get("best_log_scale", -99)
    best_s = 10**best_log_s if best_log_s > -50 else 0
    ratio = best_s / G_DAGGER if best_s > 0 else 0

    print(f"  Best kernel: {best}")
    print(f"  Best scale: 10^{best_log_s:.3f} = {best_s:.2e} m/s²")
    print(f"  Ratio to g†: {ratio:.1f}× (expected ~14×)")

    delta_from_expected = abs(best_log_s - np.log10(14 * G_DAGGER))
    print_verdict(
        "Scale near 14× g† (cluster prediction)",
        delta_from_expected < 0.3,
        f"Δ(from 14×g†) = {delta_from_expected:.3f} dex"
    )

    if "shuffle_null" in summary:
        p_rms = summary["shuffle_null"]["p_value_rms"]
        print_verdict(
            "Fit better than shuffled",
            p_rms < 0.10,
            f"p(rms) = {p_rms:.4f}"
        )

    print(f"\n  Output: {summary['output_dir']}")
    return summary


def main() -> None:
    """Run all pilot experiments."""
    parser = argparse.ArgumentParser(description="g† Hunt Pilot Tests")
    parser.add_argument("--n-shuffles", type=int, default=200,
                        help="Number of shuffles per test")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel shuffles")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test (minimal shuffles)")
    parser.add_argument("--output-base", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "outputs", "gdagger_hunt"))
    args = parser.parse_args()

    if args.smoke:
        args.n_shuffles = 20
        print("*** SMOKE TEST MODE (20 shuffles) ***")

    config = ExperimentConfig(
        seed=args.seed,
        n_shuffles=args.n_shuffles,
        parallel=args.parallel,
        n_workers=args.n_workers,
        output_base=args.output_base,
        n_grid=200 if not args.smoke else 80,
        n_scan=300 if not args.smoke else 100,
    )

    t0 = time.time()
    all_results = {}

    # Pi group demo
    run_pi_group_demo()

    # Experiment 1: Synthetic
    all_results["synthetic"] = run_synthetic_test(config)

    # Experiment 2: SPARC RAR control
    all_results["sparc_rar"] = run_rar_control(config)

    # Experiment 3: Cluster pilot
    all_results["cluster"] = run_cluster_pilot(config)

    # --- Three-panel summary figure ---
    panel_summaries = {}
    label_map = {"synthetic": "Synthetic", "sparc_rar": "SPARC RAR",
                 "cluster": "Cluster"}
    for key, result in all_results.items():
        if not result.get("skipped"):
            panel_summaries[label_map.get(key, key)] = result

    if len(panel_summaries) >= 2:
        fig_path = plot_three_panel_summary(
            panel_summaries,
            sparc_key="SPARC RAR",
            output_path=os.path.join(config.output_base,
                                     "three_panel_summary.png"),
        )
        print(f"\n  Three-panel figure: {fig_path}")

        # Print interpretation guardrails if present
        sparc_nearby = all_results.get("sparc_rar", {}).get(
            "nearby_scales", {})
        notes = sparc_nearby.get("interpretation_notes", {})
        if notes:
            print("\n  Interpretation guardrails:")
            for k, v in notes.items():
                print(f"    [{k}] {v}")

    # Final summary
    elapsed = time.time() - t0
    print_header("PILOT SUMMARY")
    print(f"  Total time: {elapsed:.1f} s")
    print(f"  Output base: {config.output_base}\n")

    n_pass = 0
    n_total = 0

    for name, result in all_results.items():
        if result.get("skipped"):
            print(f"  {name}: SKIPPED ({result.get('reason', '')})")
            continue
        verdicts = result.get("verdicts", {})
        for vk, vv in verdicts.items():
            n_total += 1
            is_pass = vv in ("STRONG_HIT", "HIT", "SIGNIFICANT", "SHARP",
                             "MODERATE", "BE_FAMILY")
            if is_pass:
                n_pass += 1
            print(f"  {name}/{vk}: {vv}")

    print(f"\n  Passed: {n_pass}/{n_total} verdicts")

    # Write combined pilot summary
    pilot_summary = {
        "elapsed_seconds": elapsed,
        "n_shuffles": config.n_shuffles,
        "seed": config.seed,
        "results": {},
    }
    for name, result in all_results.items():
        if not result.get("skipped"):
            pilot_summary["results"][name] = {
                "output_dir": result.get("output_dir", ""),
                "verdicts": result.get("verdicts", {}),
                "best_kernel": result.get("best_kernel_name", ""),
                "best_log_scale": result.get("best_log_scale", None),
            }
        else:
            pilot_summary["results"][name] = {"skipped": True}

    pilot_path = os.path.join(config.output_base, "pilot_summary.json")
    os.makedirs(os.path.dirname(pilot_path), exist_ok=True)
    with open(pilot_path, "w") as f:
        json.dump(pilot_summary, f, indent=2, default=str)
    print(f"\n  Pilot summary: {pilot_path}")


if __name__ == "__main__":
    main()
