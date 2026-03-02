#!/usr/bin/env python3
"""
Merge sharded shuffle/permutation outputs from run_referee_required_tests.py.

Usage:
    python3 merge_referee_shuffle_shards.py --output-dir <path>

Expects:
    <output-dir>/shuffle_shards/shard_<id>_of_<N>/
        shard_metadata.json
        test1_A_shuffle_results.npz
        test1_B_shuffle_results.npz
        test3_sparc_cnull.npz   (optional)
        test3_tng_cnull.npz     (optional)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(obj), indent=2))


def robust_percentiles(x: np.ndarray, q: list) -> list:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return [None for _ in q]
    return [float(v) for v in np.percentile(y, q)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge referee shuffle shards.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path containing shuffle_shards/ subdirectory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    shard_base = out_dir / "shuffle_shards"
    if not shard_base.exists():
        raise FileNotFoundError(f"No shuffle_shards/ directory found in {out_dir}")

    # Discover shards
    meta_files = sorted(shard_base.glob("shard_*_of_*/shard_metadata.json"))
    if not meta_files:
        raise FileNotFoundError("No shard_metadata.json files found")

    metas: List[Dict[str, Any]] = []
    for mf in meta_files:
        metas.append(json.loads(mf.read_text()))

    num_shards = metas[0]["num_shards"]
    n_shuffles = metas[0]["n_shuffles"]
    seed = metas[0]["seed"]

    # Validate all shards present and consistent
    found_ids = sorted([m["shard_id"] for m in metas])
    expected_ids = list(range(num_shards))
    if found_ids != expected_ids:
        missing = set(expected_ids) - set(found_ids)
        raise ValueError(f"Missing shards: {missing}. Found: {found_ids}")

    for m in metas:
        if m["num_shards"] != num_shards:
            raise ValueError(f"Inconsistent num_shards: {m['num_shards']} vs {num_shards}")
        if m["n_shuffles"] != n_shuffles:
            raise ValueError(f"Inconsistent n_shuffles: {m['n_shuffles']} vs {n_shuffles}")
        if m["seed"] != seed:
            raise ValueError(f"Inconsistent seed: {m['seed']} vs {seed}")

    shard_dirs = [mf.parent for mf in meta_files]
    delta_real = metas[0].get("delta_real")
    daic_real = metas[0].get("daic_real")
    mu_real = metas[0].get("mu_real")

    # --- Test 1: merge shuffle null distributions ---
    print("[MERGE] Test 1: phase peak null distribution")
    for key, mode_label in [("A", "A_within_galaxy"), ("B", "B_galaxy_label")]:
        all_mu: List[np.ndarray] = []
        all_delta: List[np.ndarray] = []
        all_daic: List[np.ndarray] = []
        for sd in shard_dirs:
            npz_path = sd / f"test1_{key}_shuffle_results.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing {npz_path}")
            data = np.load(npz_path)
            all_mu.append(data["mu_null"])
            all_delta.append(data["delta_null"])
            all_daic.append(data["daic_null"])

        mu_null = np.concatenate(all_mu)
        delta_null = np.concatenate(all_delta)
        daic_null = np.concatenate(all_daic)

        p_delta = float(np.mean(np.where(np.isfinite(delta_null), delta_null <= delta_real, False)))
        p_daic = float(np.mean(np.where(np.isfinite(daic_null), daic_null <= daic_real, False)))
        d50, d05, d95 = robust_percentiles(delta_null, [50, 5, 95])
        a50, a05, a95 = robust_percentiles(daic_null, [50, 5, 95])

        print(f"  {mode_label}: p_delta={p_delta:.4g}, p_daic={p_daic:.4g}, "
              f"n_null={len(delta_null)}")

        # Store for summary
        if key == "A":
            shuffle_A = {
                "p_delta": p_delta, "p_daic": p_daic,
                "delta_null_median": d50, "delta_null_5pct": d05, "delta_null_95pct": d95,
                "daic_null_median": a50, "daic_null_5pct": a05, "daic_null_95pct": a95,
                "n_fail": int(np.sum(~np.isfinite(delta_null))),
                "n_fallback": 0,
            }
        else:
            shuffle_B = {
                "p_delta": p_delta, "p_daic": p_daic,
                "delta_null_median": d50, "delta_null_5pct": d05, "delta_null_95pct": d95,
                "daic_null_median": a50, "daic_null_5pct": a05, "daic_null_95pct": a95,
                "n_fail": int(np.sum(~np.isfinite(delta_null))),
                "n_fallback": 0,
            }

    summary_t1 = {
        "test": "phase_peak_null_distribution",
        "merged_from_shards": num_shards,
        "n_shuffles": n_shuffles,
        "real": {
            "mu_peak": mu_real,
            "delta_from_gdagger": delta_real,
            "daic": daic_real,
        },
        "shuffle_A_within_galaxy": shuffle_A,
        "shuffle_B_galaxy_label": shuffle_B,
    }
    write_json(out_dir / "summary_phase_peak_null.json", summary_t1)
    print(f"  Wrote {out_dir / 'summary_phase_peak_null.json'}")

    # --- Test 3: merge permutation null distributions ---
    print("[MERGE] Test 3: xi organizing permutation null")
    summary_t3_path = out_dir / "summary_xi_organizing.json"
    t3_summary: Dict[str, Any] = {}
    if summary_t3_path.exists():
        t3_summary = json.loads(summary_t3_path.read_text())

    for tag in ["sparc", "tng"]:
        all_cnull: List[np.ndarray] = []
        any_found = False
        for sd in shard_dirs:
            npz_path = sd / f"test3_{tag}_cnull.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                all_cnull.append(data["cnull"])
                any_found = True

        if not any_found:
            print(f"  {tag}: no shard files found, skipping")
            continue

        cnull = np.concatenate(all_cnull)
        C_real_key = f"C_real_{tag}"
        C_real: Optional[float] = None
        for m in metas:
            if C_real_key in m:
                C_real = float(m[C_real_key])
                break

        if C_real is None:
            print(f"  {tag}: no C_real found in shard metadata, skipping p-value")
            continue

        p_c = float(np.mean(np.where(np.isfinite(cnull), cnull >= C_real, False)))
        c50, c5, c95 = robust_percentiles(cnull, [50, 5, 95])

        print(f"  {tag}: p_c={p_c:.4g}, C_real={C_real:.4f}, n_null={len(cnull)}")

        t3_summary[tag] = {
            "status": "OK",
            "dataset": tag.upper(),
            "concentration_C": C_real,
            "concentration_pvalue": p_c,
            "concentration_null_median": c50,
            "concentration_null_5pct": c5,
            "concentration_null_95pct": c95,
            "merged_from_shards": num_shards,
        }

    if t3_summary:
        write_json(summary_t3_path, t3_summary)
        print(f"  Wrote {summary_t3_path}")

    # Write marker file
    marker = out_dir / "MERGED_SHARDS"
    marker.write_text(
        f"Merged {num_shards} shards, seed={seed}, n_shuffles={n_shuffles}\n"
    )
    print(f"\nDone. Marker: {marker}")


if __name__ == "__main__":
    main()
